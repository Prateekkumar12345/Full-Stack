const API_BASE_URL = 'http://localhost:8000'; // Your FastAPI backend URL

export const getRecommendations = async (studentData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(studentData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get recommendations');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    throw error;
  }
};

export const getSampleRequest = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/sample-request`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching sample request:', error);
    throw error;
  }
};
