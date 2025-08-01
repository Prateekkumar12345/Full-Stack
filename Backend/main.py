import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import sentence transformers, fallback to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✅ SentenceTransformers available")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ SentenceTransformers not available, using TF-IDF fallback")

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# LLM Integration
try:
    from groq import Groq
    import json
    import re
    GROQ_AVAILABLE = True
    print("✅ Groq library available")
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq library not available, LLM fallback disabled")

print("✅ All libraries imported successfully!")

class TextEmbedder:
    """
    A wrapper class that uses SentenceTransformers if available,
    otherwise falls back to TF-IDF vectorization
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.use_sentence_transformers = False
        self.model = None
        self.tfidf = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try to load with offline mode first
                self.model = SentenceTransformer(model_name, device='cpu')
                self.use_sentence_transformers = True
                print(f"✅ Using SentenceTransformer: {model_name}")
            except Exception as e:
                print(f"⚠️ SentenceTransformer failed: {e}")
                print("🔄 Falling back to TF-IDF vectorization")
                self._setup_tfidf()
        else:
            print("🔄 Using TF-IDF vectorization")
            self._setup_tfidf()

    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer as fallback"""
        self.tfidf = TfidfVectorizer(
            max_features=384,  # Similar to sentence transformer dimension
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.use_sentence_transformers = False

    def encode(self, texts):
        """
        Encode texts using either SentenceTransformer or TF-IDF

        Parameters:
        -----------
        texts : list
            List of texts to encode

        Returns:
        --------
        numpy.ndarray
            Encoded vectors
        """
        if self.use_sentence_transformers and self.model is not None:
            try:
                return self.model.encode(texts, show_progress_bar=False)
            except Exception as e:
                print(f"⚠️ SentenceTransformer encoding failed: {e}")
                print("🔄 Switching to TF-IDF fallback")
                self._setup_tfidf()
                return self._encode_with_tfidf(texts)
        else:
            return self._encode_with_tfidf(texts)

    def _encode_with_tfidf(self, texts):
        """Encode texts using TF-IDF"""
        if not hasattr(self.tfidf, 'vocabulary_') or self.tfidf.vocabulary_ is None:
            # First time - fit the vectorizer
            vectors = self.tfidf.fit_transform(texts)
        else:
            # Already fitted - just transform
            vectors = self.tfidf.transform(texts)

        return vectors.toarray()

class StudentModel:
    def __init__(self, num_clusters=5, random_state=42):
        """
        Initialize the StudentModel for processing student data

        Parameters:
        -----------
        num_clusters : int
            Number of clusters to group students into
        random_state : int
            Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.text_embedder = TextEmbedder('all-MiniLM-L6-v2')
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.numerical_columns = ['Marks_10th', 'Marks_12th', 'JEE_Score', 'Budget']
        self.categorical_columns = [
            'Preferred Location', 'Gender', 'Target Exam', 'State Board', 'Category',
            'Stress Tolerance', 'English Proficiency'
        ]
        self.text_columns = ['Extra Curriculars', 'Future Goal', 'Certifications']
        self.fitted = False

    def preprocess_data(self, df):
        """
        Preprocess the student data by scaling numerical features and encoding categorical features

        Parameters:
        -----------
        df : pandas.DataFrame
            Student dataset

        Returns:
        --------
        tuple
            Processed numerical, categorical, and text features
        """
        # Handle numerical features
        numerical_features = df[self.numerical_columns].copy()

        # Handle categorical features
        categorical_features = df[self.categorical_columns].copy()

        # Handle text features - combine for embedding
        text_features = []
        for _, row in df.iterrows():
            text = f"{row['Extra Curriculars']} {row['Certifications']} {row['Future Goal']}"
            text_features.append(text)

        return numerical_features, categorical_features, text_features

    def fit(self, df):
        """
        Fit the model to the student data

        Parameters:
        -----------
        df : pandas.DataFrame
            Student dataset

        Returns:
        --------
        self
        """
        print("🔄 Processing student data...")

        # Preprocess data
        numerical_features, categorical_features, text_features = self.preprocess_data(df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        print(f"✅ Scaled {scaled_numerical.shape[1]} numerical features")

        # Encode categorical features
        encoded_categorical = self.categorical_encoder.fit_transform(categorical_features)
        print(f"✅ Encoded {encoded_categorical.shape[1]} categorical features")

        # Embed text features
        print("🔄 Encoding text features...")
        text_embeddings = self.text_embedder.encode(text_features)
        print(f"✅ Generated {text_embeddings.shape[1]} text embedding features")

        # Combine all features
        combined_features = np.hstack([scaled_numerical, encoded_categorical, text_embeddings])
        print(f"✅ Combined features shape: {combined_features.shape}")

        # Apply PCA for dimensionality reduction
        self.pca_features = self.pca.fit_transform(combined_features)
        print(f"✅ PCA reduced to {self.pca_features.shape[1]} components")

        # Apply KMeans clustering
        self.cluster_labels = self.kmeans.fit_predict(self.pca_features)
        print(f"✅ KMeans clustering completed")

        # Store original data
        self.original_data = df.copy()

        # Add cluster labels to original data
        self.original_data['Cluster'] = self.cluster_labels

        self.fitted = True
        return self

    def predict_cluster(self, student_dict):
        """
        Predict the cluster for a new student

        Parameters:
        -----------
        student_dict : dict
            Dictionary containing student information

        Returns:
        --------
        tuple
            Predicted cluster ID and PCA features
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Convert dictionary to DataFrame
        student_df = pd.DataFrame([student_dict])

        # Preprocess the new student data
        numerical_features, categorical_features, text_features = self.preprocess_data(student_df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.transform(numerical_features)

        # Encode categorical features
        encoded_categorical = self.categorical_encoder.transform(categorical_features)

        # Embed text features
        text_embeddings = self.text_embedder.encode(text_features)

        # Combine all features
        combined_features = np.hstack([scaled_numerical, encoded_categorical, text_embeddings])

        # Apply PCA transformation
        pca_features = self.pca.transform(combined_features)

        # Predict cluster
        cluster = self.kmeans.predict(pca_features)[0]

        return cluster, pca_features

    def get_students_in_cluster(self, cluster_id):
        """
        Get all students in a specific cluster

        Parameters:
        -----------
        cluster_id : int
            Cluster ID

        Returns:
        --------
        pandas.DataFrame
            Students in the specified cluster
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.original_data[self.original_data['Cluster'] == cluster_id]

    def visualize_clusters(self):
        """
        Visualize the student clusters in 2D PCA space
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Use first two PCA components for visualization
        pca_2d = PCA(n_components=2)
        pca_result_2d = pca_2d.fit_transform(self.pca_features)

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_result_2d[:, 0], y=pca_result_2d[:, 1], hue=self.cluster_labels, palette='viridis')
        plt.title('Student Clusters in 2D PCA Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cluster')
        plt.show()

print("✅ StudentModel class defined successfully!")

class CollegeModel:
    def __init__(self, num_clusters=8, random_state=42):
        """
        Initialize the CollegeModel for processing college data

        Parameters:
        -----------
        num_clusters : int
            Number of clusters to group colleges into
        random_state : int
            Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.text_embedder = TextEmbedder('all-MiniLM-L6-v2')
        self.numerical_scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        self.fitted = False

    def preprocess_data(self, df):
        """
        Preprocess the college data by creating a textual representation and extracting numerical features

        Parameters:
        -----------
        df : pandas.DataFrame
            College dataset

        Returns:
        --------
        tuple
            Processed text features and numerical features
        """
        # Extract important numerical features
        numerical_features = []

        # Extract rankings and convert to numeric
        nirf_ranking = pd.to_numeric(df['NIRF Ranking'], errors='coerce')
        numerical_features.append(nirf_ranking)

        # Extract and process fees (assuming it's in LPA format)
        fees_data = []
        for fee in df['Course Fees (₹)']:
            if isinstance(fee, str) and 'LPA' in fee:
                try:
                    # Extract numeric part before LPA
                    value = float(fee.replace('LPA', '').replace('₹', '').strip())
                    fees_data.append(value)
                except:
                    fees_data.append(np.nan)
            else:
                fees_data.append(np.nan)

        numerical_features.append(fees_data)

        # Convert to numpy array and transpose
        numerical_features = np.array(numerical_features).T

        # Replace NaN values with column means
        for col in range(numerical_features.shape[1]):
            col_mean = np.nanmean(numerical_features[:, col])
            numerical_features[:, col] = np.nan_to_num(numerical_features[:, col], nan=col_mean)

        # Create textual representation for each college
        text_features = []
        for _, row in df.iterrows():
            college_text = f"College {row['College Name']} located in {row['Location']}, {row['State']} "
            college_text += f"is a {row['College Type']} established in {row['Established Year']}. "
            college_text += f"It is approved by {row['Approved By']} with NIRF ranking {row['NIRF Ranking']}. "
            college_text += f"The college offers courses in {row['Notable Courses Offered']}. "

            # Add information about education loan and placement
            if 'Education Loan' in row and row['Education Loan'] == 'Yes':
                college_text += "Education loan facility is available. "

            if 'Placement (Average' in row and not pd.isna(row['Placement (Average']):
                college_text += f"Average placement is {row['Placement (Average']}. "

            text_features.append(college_text)

        return text_features, numerical_features

    def fit(self, df):
        """
        Fit the model to the college data

        Parameters:
        -----------
        df : pandas.DataFrame
            College dataset

        Returns:
        --------
        self
        """
        print("🔄 Processing college data...")

        # Preprocess data
        text_features, numerical_features = self.preprocess_data(df)

        # Scale numerical features
        scaled_numerical = self.numerical_scaler.fit_transform(numerical_features)
        print(f"✅ Scaled {scaled_numerical.shape[1]} numerical features")

        # Embed text features
        print("🔄 Encoding college text features...")
        text_embeddings = self.text_embedder.encode(text_features)
        print(f"✅ Generated {text_embeddings.shape[1]} text embedding features")

        # Combine all features
        combined_features = np.hstack([scaled_numerical, text_embeddings])
        print(f"✅ Combined features shape: {combined_features.shape}")

        # Check for NaN values before PCA
        if np.isnan(combined_features).any():
            print("⚠️ Warning: NaN values found in combined features, replacing with zeros")
            combined_features = np.nan_to_num(combined_features)

        # Apply PCA for dimensionality reduction
        self.pca_features = self.pca.fit_transform(combined_features)
        print(f"✅ PCA reduced to {self.pca_features.shape[1]} components")

        # Apply KMeans clustering
        self.cluster_labels = self.kmeans.fit_predict(self.pca_features)
        print(f"✅ KMeans clustering completed")

        # Store original data
        self.original_data = df.copy()

        # Add cluster labels and PCA features to original data
        self.original_data['Cluster'] = self.cluster_labels
        for i in range(min(5, self.pca_features.shape[1])):  # Store first 5 PCA features
            self.original_data[f'PCA_{i+1}'] = self.pca_features[:, i]

        self.fitted = True
        return self

    def get_colleges_in_cluster(self, cluster_id):
        """
        Get all colleges in a specific cluster

        Parameters:
        -----------
        cluster_id : int
            Cluster ID

        Returns:
        --------
        pandas.DataFrame
            Colleges in the specified cluster
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.original_data[self.original_data['Cluster'] == cluster_id]

    def get_pca_features(self):
        """
        Get the PCA features for all colleges

        Returns:
        --------
        numpy.ndarray
            PCA features
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        return self.pca_features

    def visualize_clusters(self):
        """
        Visualize the college clusters in 2D PCA space
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Use first two PCA components for visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=self.pca_features[:, 0],
            y=self.pca_features[:, 1],
            hue=self.cluster_labels,
            palette='viridis',
            s=100,
            alpha=0.7
        )

        plt.title('College Clusters in 2D PCA Space')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(title='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

print("✅ CollegeModel class defined successfully!")

class Recommender:
    def __init__(self, student_model, college_model):
        """
        Initialize the recommender with student and college models

        Parameters:
        -----------
        student_model : StudentModel
            Trained student model
        college_model : CollegeModel
            Trained college model
        """
        self.student_model = student_model
        self.college_model = college_model

    def recommend(self, student_dict, top_n=5):
        # Predict student cluster and get PCA features
        student_cluster, student_pca = self.student_model.predict_cluster(student_dict)

        # Get all colleges data with PCA features
        colleges_data = self.college_model.original_data
        college_pca_features = self.college_model.pca_features

        # Calculate similarity
        n_components = min(5, student_pca.shape[1], college_pca_features.shape[1])
        student_pca_truncated = student_pca[0, :n_components].reshape(1, -1)
        college_pca_truncated = college_pca_features[:, :n_components]
        similarities = cosine_similarity(student_pca_truncated, college_pca_truncated)[0]

        # Add similarity scores
        colleges_with_scores = colleges_data.copy()
        colleges_with_scores['Similarity'] = similarities

        # Filter by preferences
        filtered_colleges = self._filter_by_preferences(colleges_with_scores, student_dict)

        # Split into preferred location and others
        if 'Preferred Location' in student_dict and student_dict['Preferred Location']:
            location_pref = student_dict['Preferred Location'].strip().lower()
            preferred_mask = (
                (filtered_colleges['State'].str.lower() == location_pref) |
                (filtered_colleges['Location'].str.lower() == location_pref)
            )
            preferred_colleges = filtered_colleges[preferred_mask]
            other_colleges = filtered_colleges[~preferred_mask]
        else:
            preferred_colleges = pd.DataFrame()
            other_colleges = filtered_colleges

        # Get recommendations - top 3 from preferred, top 2 from others
        pref_rec = preferred_colleges.sort_values('Similarity', ascending=False).head(3)
        other_rec = other_colleges.sort_values('Similarity', ascending=False).head(2)

        # Combine recommendations
        recommendations = pd.concat([pref_rec, other_rec]).head(top_n)

        # If we don't have enough preferred colleges, fill with others
        if len(recommendations) < top_n:
            additional_needed = top_n - len(recommendations)
            additional = other_colleges.sort_values('Similarity', ascending=False).head(additional_needed)
            recommendations = pd.concat([recommendations, additional]).head(top_n)

        return recommendations

    def _filter_by_preferences(self, colleges_df, student_dict):
        filtered_df = colleges_df.copy()

        # Budget filtering with priority to closer matches
        if 'Budget' in student_dict and student_dict['Budget'] > 0:
            budget = student_dict['Budget']

            def get_fee_value(fee_str):
                if pd.isna(fee_str) or not isinstance(fee_str, str):
                    return np.nan
                try:
                    if 'LPA' in fee_str:
                        return float(fee_str.replace('LPA', '').replace('₹', '').strip()) * 100000
                    else:
                        return float(''.join(filter(str.isdigit, fee_str)))
                except:
                    return np.nan

            # Calculate fee values and differences from budget
            filtered_df['Fee_Value'] = filtered_df['Course Fees (₹)'].apply(get_fee_value)
            filtered_df['Budget_Diff'] = abs(filtered_df['Fee_Value'] - budget)

            # Filter out colleges way over budget (more than 20% over)
            filtered_df = filtered_df[
                (filtered_df['Fee_Value'] <= budget * 1.2) |
                (filtered_df['Fee_Value'].isna())
            ]

            # Sort by budget difference (closest to budget first)
            filtered_df = filtered_df.sort_values('Budget_Diff', ascending=True)

        # Location filtering
        if 'Preferred Location' in student_dict and student_dict['Preferred Location']:
            location_pref = student_dict['Preferred Location'].strip().lower()
            if location_pref:
                # Exact match for state or location
                location_mask = (
                    (filtered_df['State'].str.lower() == location_pref) |
                    (filtered_df['Location'].str.lower() == location_pref)
                )

                # If no exact matches, try partial matches
                if location_mask.sum() == 0:
                    location_mask = (
                        filtered_df['State'].str.lower().str.contains(location_pref, na=False) |
                        filtered_df['Location'].str.lower().str.contains(location_pref, na=False)
                    )

                filtered_df = filtered_df[location_mask]

        return filtered_df

print("✅ Recommender class defined successfully!")

# Create sample data function
def create_sample_data():
    """Create sample datasets if real files are not available"""

    print("🔄 Creating sample student data...")
    student_data = pd.DataFrame({
        'Marks_10th': np.random.normal(80, 10, 100).clip(60, 100),
        'Marks_12th': np.random.normal(82, 10, 100).clip(60, 100),
        'JEE_Score': np.random.normal(120, 30, 100).clip(0, 300),
        'Budget': np.random.normal(500000, 200000, 100).clip(100000, 2000000),
        'Preferred Location': np.random.choice(['Karnataka', 'Delhi', 'Maharashtra', 'Tamil Nadu'], 100),
        'Gender': np.random.choice(['Male', 'Female'], 100),
        'Target Exam': np.random.choice(['JEE', 'NEET', 'CUET'], 100),
        'State Board': np.random.choice(['CBSE', 'ICSE', 'State Board'], 100),
        'Category': np.random.choice(['General', 'OBC', 'SC', 'ST'], 100),
        'Stress Tolerance': np.random.choice(['Low', 'Average', 'High'], 100),
        'English Proficiency': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], 100),
        'Extra Curriculars': np.random.choice(['Sports, Music', 'Debate, Drama', 'Coding, Robotics', 'Art, Photography'], 100),
        'Future Goal': np.random.choice(['Engineering Career', 'Medical Career', 'Management Career', 'Research Career'], 100),
        'Certifications': np.random.choice(['Programming', 'Data Science', 'Web Development', 'Digital Marketing'], 100)
    })

    print("🔄 Creating sample college data...")
    college_data = pd.DataFrame({
        'College Name': [
            'Delhi Technological University', 'Indian Institute of Technology Delhi',
            'Netaji Subhas University of Technology', 'Jamia Millia Islamia',
            'University of Delhi', 'Indian Institute of Technology Bombay',
            'Veermata Jijabai Technological Institute', 'College of Engineering Pune',
            'Bangalore Institute of Technology', 'Chennai Institute of Technology'
        ] * 5,
        'Location': np.random.choice(['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Pune', 'Hyderabad'], 50),
        'State': np.random.choice(['Karnataka', 'Delhi', 'Maharashtra', 'Tamil Nadu', 'Telangana'], 50),
        'College Type': np.random.choice(['Government', 'Private', 'Autonomous'], 50),
        'Established Year': np.random.randint(1950, 2020, 50),
        'Approved By': np.random.choice(['AICTE', 'UGC', 'NBA'], 50),
        'NIRF Ranking': np.random.randint(1, 200, 50),
        'Course Fees (₹)': [f'{np.random.uniform(2, 10):.1f} LPA' for _ in range(50)],
        'Notable Courses Offered': np.random.choice([
            'Engineering, Management', 'Engineering, Science', 'Management, Commerce',
            'Engineering, Technology', 'Science, Research'
        ], 50),
        'Education Loan': np.random.choice(['Yes', 'No'], 50),
        'Placement (Average': [f'{np.random.uniform(5, 15):.1f} LPA' for _ in range(50)]
    })

    return student_data, college_data

# Global variables for loaded models
student_model = None
college_model = None
recommender = None
groq_client = None

def initialize_groq():
    """Initialize Groq client for LLM fallback"""
    global groq_client
    
    if not GROQ_AVAILABLE:
        print("⚠️ Groq library not available")
        return False
    
    # Set your GROQ API key directly here or from environment
    api_key = ""
    
    # Fallback to environment variables
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY') or os.getenv('groq22')

    if not api_key:
        print("⚠️ GROQ_API_KEY not found")
        return False
    
    try:
        groq_client = Groq(api_key=api_key)
        print("✅ Groq client initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing Groq client: {e}")
        return False

def load_models():
    """Load the trained models from pickle files"""
    global student_model, college_model, recommender
    
    models_dir = "models"
    
    try:
        # Load student model
        with open(os.path.join(models_dir, "student_model.pkl"), 'rb') as f:
            student_model = pickle.load(f)
        print("✅ Student model loaded successfully!")

        # Load college model
        with open(os.path.join(models_dir, "college_model.pkl"), 'rb') as f:
            college_model = pickle.load(f)
        print("✅ College model loaded successfully!")

        # Load recommender
        with open(os.path.join(models_dir, "recommender.pkl"), 'rb') as f:
            recommender = pickle.load(f)
        print("✅ Recommender loaded successfully!")

        # Load metadata
        try:
            with open(os.path.join(models_dir, "metadata.pkl"), 'rb') as f:
                metadata = pickle.load(f)
            print(f"✅ Model metadata loaded: {metadata}")
        except FileNotFoundError:
            print("⚠️ Metadata file not found")

        return True

    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("🔄 Please run the training script first to create the models")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def train_and_save_models():
    """Train the models and save them"""
    print("=" * 60)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 60)

    # Load datasets
    try:
        student_data = pd.read_excel("student_dataset00.xlsx")
        print(f"✅ Student dataset loaded: {student_data.shape[0]} records, {student_data.shape[1]} features")
    except Exception as e:
        print(f"⚠️ Could not load student_dataset00.xlsx: {e}")
        student_data, _ = create_sample_data()
        print(f"✅ Created sample student data: {student_data.shape}")

    try:
        college_data = pd.read_excel("210colleges_dataset_krip.ai.xlsx")
        print(f"✅ College dataset loaded: {college_data.shape[0]} records, {college_data.shape[1]} features")
    except Exception as e:
        print(f"⚠️ Could not load 210colleges_dataset_krip.ai.xlsx: {e}")
        _, college_data = create_sample_data()
        print(f"✅ Created sample college data: {college_data.shape}")

    print("\n=== TRAINING MODELS ===")

    # Train Student Model
    print("\n1. Training Student Model...")
    start_time = datetime.now()
    global student_model, college_model, recommender
    student_model = StudentModel(num_clusters=5, random_state=42)
    student_model.fit(student_data)
    training_time = datetime.now() - start_time
    print(f"✅ Student model trained in {training_time.total_seconds():.2f} seconds")

    # Train College Model
    print("\n2. Training College Model...")
    start_time = datetime.now()
    college_model = CollegeModel(num_clusters=8, random_state=42)
    college_model.fit(college_data)
    training_time = datetime.now() - start_time
    print(f"✅ College model trained in {training_time.total_seconds():.2f} seconds")

    # Create Recommender
    print("\n3. Creating Recommender System...")
    recommender = Recommender(student_model, college_model)
    print("✅ Recommender system created successfully!")

    # Test the system
    print("\n=== TESTING RECOMMENDATION SYSTEM ===")
    test_student = {
        'Marks_10th': 85,
        'Marks_12th': 82,
        'JEE_Score': 125,
        'Budget': 350000,
        'Preferred Location': 'Karnataka',
        'Gender': 'Male',
        'Certifications': 'Python, AI',
        'Target Exam': 'JEE',
        'State Board': 'CBSE',
        'Category': 'General',
        'Stress Tolerance': 'High',
        'English Proficiency': 'Excellent',
        'Extra Curriculars': 'Coding, Sports',
        'Future Goal': 'AI Engineer'
    }

    try:
        recommendations = recommender.recommend(test_student, top_n=5)
        print(f"✅ Generated {len(recommendations)} recommendations")

        print(f"\nTop Recommendations:")
        for i, (_, row) in enumerate(recommendations.iterrows()):
            print(f"{i+1}. {row['College Name']} - Similarity: {row['Similarity']:.3f}")

    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")

    # Save models
    print("\n=== SAVING MODELS ===")
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    try:
        # Save models
        with open(os.path.join(models_dir, "student_model.pkl"), 'wb') as f:
            pickle.dump(student_model, f)
        print("✅ Student model saved")

        with open(os.path.join(models_dir, "college_model.pkl"), 'wb') as f:
            pickle.dump(college_model, f)
        print("✅ College model saved")

        with open(os.path.join(models_dir, "recommender.pkl"), 'wb') as f:
            pickle.dump(recommender, f)
        print("✅ Recommender saved")

        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'student_clusters': student_model.num_clusters,
            'college_clusters': college_model.num_clusters,
            'num_students': len(student_data),
            'num_colleges': len(college_data),
            'text_embedding_method': 'SentenceTransformer' if student_model.text_embedder.use_sentence_transformers else 'TF-IDF'
        }

        with open(os.path.join(models_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        print("✅ Metadata saved")

        return True

    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def get_llm_recommendations(student_profile):
    """Get college recommendations from LLM when model fails"""
    if not groq_client:
        return None
    
    # Create a detailed prompt for the LLM
    prompt = f"""
You are an expert college counselor in India. Based on the following student profile, recommend 5 specific colleges/universities that would be the best fit. For each recommendation, provide:

1. College Name
2. Location (City, State)  
3. College Type (Government/Private/Deemed)
4. Approximate Annual Fees (in ₹)
5. Why it's a good fit (brief explanation)

Student Profile:
- 10th Marks: {student_profile['Marks_10th']}%
- 12th Marks: {student_profile['Marks_12th']}%
- Entrance Exam Score: {student_profile['JEE_Score']} (Exam: {student_profile['Target Exam']})
- Budget: ₹{student_profile['Budget']} per year
- Preferred Location: {student_profile['Preferred Location']}
- Category: {student_profile['Category']}
- Education Board: {student_profile['State Board']}
- Career Goals: {student_profile['Future Goal']}
- Certifications: {student_profile['Certifications']}
- Extra-curriculars: {student_profile['Extra Curriculars']}
- English Proficiency: {student_profile['English Proficiency']}
- Stress Tolerance: {student_profile['Stress Tolerance']}

Please provide realistic recommendations based on the student's scores and budget. Focus on colleges that actually exist in India and match the student's academic profile.

Format your response as a JSON array with this structure:
[
  {{
    "rank": 1,
    "college_name": "College Name",
    "location": "City, State",
    "college_type": "Government/Private/Deemed",
    "annual_fees": 150000,
    "match_score": 0.85,
    "reason": "Brief explanation why this is a good fit"
  }}
]
"""

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Indian college counselor with deep knowledge of engineering, medical, and other professional colleges across India. Provide accurate, realistic recommendations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=2000
        )
        
        llm_response = response.choices[0].message.content
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            recommendations = json.loads(json_str)
            return recommendations
        else:
            # If JSON parsing fails, try to parse the text response
            return parse_text_response(llm_response)
            
    except Exception as e:
        print(f"Error getting LLM recommendations: {e}")
        return None

def parse_text_response(text_response):
    """Parse text response if JSON parsing fails"""
    try:
        # Simple parsing for text responses
        recommendations = []
        lines = text_response.split('\n')
        current_rec = {}
        rank = 1
        
        for line in lines:
            line = line.strip()
            if 'college' in line.lower() and 'name' in line.lower():
                if current_rec:
                    current_rec['rank'] = rank
                    recommendations.append(current_rec)
                    rank += 1
                current_rec = {'college_name': line.split(':')[-1].strip()}
            elif 'location' in line.lower():
                current_rec['location'] = line.split(':')[-1].strip()
            elif 'type' in line.lower():
                current_rec['college_type'] = line.split(':')[-1].strip()
            elif 'fees' in line.lower() or 'cost' in line.lower():
                fee_text = line.split(':')[-1].strip()
                # Extract numbers from fee text
                numbers = re.findall(r'\d+', fee_text)
                if numbers:
                    current_rec['annual_fees'] = int(numbers[0])
                else:
                    current_rec['annual_fees'] = 200000  # Default
            elif 'reason' in line.lower() or 'fit' in line.lower():
                current_rec['reason'] = line.split(':')[-1].strip()
        
        if current_rec:
            current_rec['rank'] = rank
            recommendations.append(current_rec)
        
        # Add default values for missing fields
        for rec in recommendations:
            rec.setdefault('match_score', 0.75)
            rec.setdefault('college_type', 'Private')
            rec.setdefault('annual_fees', 200000)
            rec.setdefault('reason', 'Good academic fit')
        
        return recommendations[:5]  # Return top 5
        
    except Exception as e:
        print(f"Error parsing text response: {e}")
        return None

# FastAPI Pydantic Models
class StudentProfile(BaseModel):
    marks_10th: float = Field(..., ge=0, le=100, description="10th standard marks percentage")
    marks_12th: float = Field(..., ge=0, le=100, description="12th standard marks percentage")
    jee_score: int = Field(..., ge=0, le=300, description="Entrance exam score")
    budget: int = Field(..., ge=50000, le=5000000, description="Annual budget in INR")
    preferred_location: str = Field(..., description="Preferred state/city")
    gender: str = Field(..., description="Gender")
    certifications: str = Field(..., description="Certifications and skills")
    target_exam: str = Field(..., description="Target entrance exam")
    state_board: str = Field(..., description="Education board")
    category: str = Field(..., description="Reservation category")
    stress_tolerance: str = Field(..., description="Stress tolerance level")
    english_proficiency: str = Field(..., description="English proficiency level")
    extra_curriculars: str = Field(..., description="Extra curricular activities")
    future_goal: str = Field(..., description="Career goals and aspirations")

class CollegeRecommendation(BaseModel):
    rank: int
    college_name: str
    location: str
    college_type: str
    fees: str
    similarity_score: float
    reason: Optional[str] = None

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[CollegeRecommendation]
    source: str  # "model" or "llm"
    message: str

class ModelStatus(BaseModel):
    student_model_loaded: bool
    college_model_loaded: bool
    recommender_loaded: bool
    groq_available: bool
    models_directory_exists: bool

# Initialize FastAPI app
app = FastAPI(
    title="College Recommendation System API",
    description="AI-powered college recommendation system with ML models and LLM fallback",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on app startup"""
    print("🚀 Starting College Recommendation System API...")
    
    # Initialize Groq client
    initialize_groq()
    
    # Try to load existing models
    models_loaded = load_models()
    
    if not models_loaded:
        print("🔄 Models not found. Training new models...")
        training_success = train_and_save_models()
        
        if training_success:
            print("✅ Training completed. Loading trained models...")
            load_models()
        else:
            print("❌ Training failed. API will use LLM fallback only.")
    
    print("✅ API startup completed!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "College Recommendation System API",
        "version": "1.0.0",
        "endpoints": {
            "/recommend": "POST - Get college recommendations",
            "/status": "GET - Check system status",
            "/health": "GET - Health check",
            "/retrain": "POST - Retrain models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status", response_model=ModelStatus)
async def get_status():
    """Get system status"""
    return ModelStatus(
        student_model_loaded=student_model is not None,
        college_model_loaded=college_model is not None,
        recommender_loaded=recommender is not None,
        groq_available=groq_client is not None,
        models_directory_exists=os.path.exists("models")
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(student_profile: StudentProfile):
    """Get college recommendations for a student"""
    
    # Convert Pydantic model to dict
    student_dict = {
        'Marks_10th': student_profile.marks_10th,
        'Marks_12th': student_profile.marks_12th,
        'JEE_Score': student_profile.jee_score,
        'Budget': student_profile.budget,
        'Preferred Location': student_profile.preferred_location,
        'Gender': student_profile.gender,
        'Certifications': student_profile.certifications,
        'Target Exam': student_profile.target_exam,
        'State Board': student_profile.state_board,
        'Category': student_profile.category,
        'Stress Tolerance': student_profile.stress_tolerance,
        'English Proficiency': student_profile.english_proficiency,
        'Extra Curriculars': student_profile.extra_curriculars,
        'Future Goal': student_profile.future_goal
    }

    try:
        # First, try the trained model
        if recommender is not None:
            recommendations_df = recommender.recommend(student_dict, top_n=5)
            
            if len(recommendations_df) > 0:
                # Format model recommendations
                recommendations = []
                for i, (_, row) in enumerate(recommendations_df.iterrows()):
                    recommendations.append(CollegeRecommendation(
                        rank=i + 1,
                        college_name=str(row['College Name']),
                        location=f"{row['Location']}, {row['State']}",
                        college_type=str(row['College Type']),
                        fees=str(row['Course Fees (₹)']),
                        similarity_score=float(row['Similarity']),
                        reason=f"Match score: {row['Similarity']:.3f} - Good fit based on your profile"
                    ))
                
                return RecommendationResponse(
                    success=True,
                    recommendations=recommendations,
                    source="model",
                    message=f"Generated {len(recommendations)} recommendations using trained ML models"
                )
        
        # If model fails or no recommendations, try LLM fallback
        print("🤖 Trying LLM fallback...")
        llm_recommendations = get_llm_recommendations(student_dict)
        
        if llm_recommendations:
            recommendations = []
            for rec in llm_recommendations:
                recommendations.append(CollegeRecommendation(
                    rank=rec.get('rank', 1),
                    college_name=rec.get('college_name', 'Unknown College'),
                    location=rec.get('location', 'Unknown Location'),
                    college_type=rec.get('college_type', 'Private'),
                    fees=f"₹{rec.get('annual_fees', 200000):,}",
                    similarity_score=rec.get('match_score', 0.75),
                    reason=rec.get('reason', 'AI-generated recommendation')
                ))
            
            return RecommendationResponse(
                success=True,
                recommendations=recommendations,
                source="llm",
                message=f"Generated {len(recommendations)} AI-powered recommendations"
            )
        
        # No recommendations available
        raise HTTPException(
            status_code=404,
            detail="No suitable colleges found for your profile. Try adjusting your criteria."
        )

    except Exception as e:
        print(f"Error in recommendation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.post("/retrain")
async def retrain_models():
    """Retrain the models with fresh data"""
    try:
        print("🔄 Starting model retraining...")
        training_success = train_and_save_models()
        
        if training_success:
            # Reload the models
            load_models()
            return {
                "success": True,
                "message": "Models retrained and reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Model training failed",
                "timestamp": datetime.now().isoformat()
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retraining models: {str(e)}"
        )

@app.get("/sample-request")
async def get_sample_request():
    """Get a sample request for testing"""
    return {
        "sample_request": {
            "marks_10th": 85.0,
            "marks_12th": 82.0,
            "jee_score": 125,
            "budget": 350000,
            "preferred_location": "Karnataka",
            "gender": "Male",
            "certifications": "Python, AI, Web Development",
            "target_exam": "JEE",
            "state_board": "CBSE",
            "category": "General",
            "stress_tolerance": "High",
            "english_proficiency": "Excellent",
            "extra_curriculars": "Coding, Sports, Music",
            "future_goal": "AI Engineer at a top tech company"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting College Recommendation System with FastAPI...")
    uvicorn.run(
        "fastapi_recommendation_system:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
