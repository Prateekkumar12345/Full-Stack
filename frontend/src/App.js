import React from 'react';
import RecommendationForm from './components/RecommendationForm';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>College Recommendation System</h1>
      </header>
      <main>
        <RecommendationForm />
      </main>
    </div>
  );
}

export default App;