import React, { useState } from "react";
import { getRecommendations, getSampleRequest } from "../services/api";

const RecommendationForm = () => {
  const [formData, setFormData] = useState({
    marks_10th: "",
    marks_12th: "",
    jee_score: "",
    budget: "",
    preferred_location: "",
    gender: "",
    certifications: "",
    target_exam: "",
    state_board: "",
    category: "",
    stress_tolerance: "",
    english_proficiency: "",
    extra_curriculars: "",
    future_goal: "",
  });

  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const data = await getRecommendations(formData);
      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message || "Failed to get recommendations");
    } finally {
      setLoading(false);
    }
  };

  const loadSampleData = async () => {
    try {
      const { sample_request } = await getSampleRequest();
      setFormData(sample_request);
    } catch (err) {
      console.error("Error loading sample data:", err);
    }
  };

  return (
    <div className="recommendation-form">
      <h2>College Recommendation System</h2>

      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3>Academic Information</h3>
          <div className="form-row">
            <div className="form-group">
              <label>10th Marks (%)</label>
              <input
                type="number"
                name="marks_10th"
                value={formData.marks_10th}
                onChange={handleChange}
                min="0"
                max="100"
                required
              />
            </div>

            <div className="form-group">
              <label>12th Marks (%)</label>
              <input
                type="number"
                name="marks_12th"
                value={formData.marks_12th}
                onChange={handleChange}
                min="0"
                max="100"
                required
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>JEE Score</label>
              <input
                type="number"
                name="jee_score"
                value={formData.jee_score}
                onChange={handleChange}
                min="0"
                max="300"
                required
              />
            </div>

            <div className="form-group">
              <label>Budget (₹)</label>
              <input
                type="number"
                name="budget"
                value={formData.budget}
                onChange={handleChange}
                min="50000"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label>Target Exam</label>
            <select
              name="target_exam"
              value={formData.target_exam}
              onChange={handleChange}
              required
            >
              <option value="JEE">JEE</option>
              <option value="NEET">NEET</option>
              <option value="CUET">CUET</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="form-group">
            <label>State Board</label>
            <select
              name="state_board"
              value={formData.state_board}
              onChange={handleChange}
              required
            >
              <option value="CBSE">CBSE</option>
              <option value="ICSE">ICSE</option>
              <option value="State Board">State Board</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>

        <div className="form-section">
          <h3>Personal Information</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                required
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>

            <div className="form-group">
              <label>Category</label>
              <select
                name="category"
                value={formData.category}
                onChange={handleChange}
                required
              >
                <option value="General">General</option>
                <option value="OBC">OBC</option>
                <option value="SC">SC</option>
                <option value="ST">ST</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Preferred Location (State/City)</label>
            <input
              type="text"
              name="preferred_location"
              value={formData.preferred_location}
              onChange={handleChange}
              required
            />
          </div>
        </div>

        <div className="form-section">
          <h3>Skills & Preferences</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Stress Tolerance</label>
              <select
                name="stress_tolerance"
                value={formData.stress_tolerance}
                onChange={handleChange}
                required
              >
                <option value="Low">Low</option>
                <option value="Average">Average</option>
                <option value="High">High</option>
              </select>
            </div>

            <div className="form-group">
              <label>English Proficiency</label>
              <select
                name="english_proficiency"
                value={formData.english_proficiency}
                onChange={handleChange}
                required
              >
                <option value="Poor">Poor</option>
                <option value="Average">Average</option>
                <option value="Good">Good</option>
                <option value="Excellent">Excellent</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label>Certifications (comma separated)</label>
            <input
              type="text"
              name="certifications"
              value={formData.certifications}
              onChange={handleChange}
              placeholder="e.g., Python, AI, Web Development"
            />
          </div>

          <div className="form-group">
            <label>Extra Curricular Activities</label>
            <input
              type="text"
              name="extra_curriculars"
              value={formData.extra_curriculars}
              onChange={handleChange}
              placeholder="e.g., Sports, Music, Debate"
            />
          </div>

          <div className="form-group">
            <label>Future Career Goal</label>
            <input
              type="text"
              name="future_goal"
              value={formData.future_goal}
              onChange={handleChange}
              placeholder="e.g., AI Engineer, Doctor, Researcher"
              required
            />
          </div>
        </div>

        <div className="form-buttons">
          <button type="button" onClick={loadSampleData} className="secondary">
            Load Sample Data
          </button>
          <button type="submit" disabled={loading}>
            {loading ? "Getting Recommendations..." : "Get Recommendations"}
          </button>
        </div>
      </form>

      {error && <div className="error-message">{error}</div>}

      {recommendations.length > 0 && (
        <div className="recommendations">
          <h3>Recommended Colleges</h3>
          <div className="recommendation-list">
            {recommendations.map((college) => (
              <div key={college.rank} className="recommendation-card">
                <h4>{college.college_name}</h4>
                <p>
                  <strong>Location:</strong> {college.location}
                </p>
                <p>
                  <strong>Type:</strong> {college.college_type}
                </p>
                <p>
                  <strong>Fees:</strong> {college.fees}
                </p>
                <p>
                  <strong>Match Score:</strong>{" "}
                  {college.similarity_score.toFixed(3)}
                </p>
                {college.reason && (
                  <p>
                    <em>{college.reason}</em>
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RecommendationForm;
