import { useState, useEffect } from 'react';
import "./App.css";

function App() {
    const [formData, setFormData] = useState({
        players: '',
        equipment: '',
        playingArea: '',
        scoringMethod: ''
    });
    const [result, setResult] = useState('');
    const [explanation, setExplanation] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [typingEffect, setTypingEffect] = useState('');
    const [typingIndex, setTypingIndex] = useState(0);
    const [confidence, setConfidence] = useState(null);

    const welcomeText = "Enter your preferences to get a personalized sport recommendation.";

    useEffect(() => {
        if (typingIndex < welcomeText.length) {
            const timeout = setTimeout(() => {
                setTypingEffect(prev => prev + welcomeText[typingIndex]);
                setTypingIndex(typingIndex + 1);
            }, 30);
            
            return () => clearTimeout(timeout);
        }
    }, [typingIndex]);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult('');
        setError('');
        setExplanation('');
        try {
            await new Promise(resolve => setTimeout(resolve, 800));
            
            const endpoint = 'http://localhost:5000/api/prescriptive';
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    players: parseInt(formData.players),
                    equipment: formData.equipment,
                    playing_area: formData.playingArea,
                    scoring_method: formData.scoringMethod
                }),
            });
            
            const data = await response.json();
            console.log(data); // Log the response to inspect its structure
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get recommendation');
            }
    
            setResult(data.recommended_sport);
            setConfidence(data.confidence || null);
            setExplanation(data.prescription || ""); // Handle prescription for the endpoint
    
        } catch (err) {
            console.error('Error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    
    return (
        <div className="container">
            <header>
                <h1>Rule-Based<span className="highlight"> Classifier</span></h1>
            </header>
            
            <main>
                <div className="content">
                    <h2>Sport Recommendation</h2>
                    <p className="typing-effect">{typingEffect}<span className="cursor">|</span></p>
                    
                    <form onSubmit={handleSubmit} className="form">
                        <div className="form-group">
                            <label htmlFor="players">Number of Players</label>
                            <input
                                type="number"
                                id="players"
                                name="players"
                                value={formData.players}
                                onChange={handleInputChange}
                                placeholder="Enter number of players"
                                required
                                className="form-input"
                            />
                        </div>
    
                        <div className="form-group">
                            <label htmlFor="equipment">Equipment</label>
                            <input
                                type="text"
                                id="equipment"
                                name="equipment"
                                value={formData.equipment}
                                onChange={handleInputChange}
                                placeholder="e.g. Ball, Racket, None"
                                required
                                className="form-input"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label htmlFor="playingArea">Playing Area</label>
                            <input
                                type="text"
                                id="playingArea"
                                name="playingArea"
                                value={formData.playingArea}
                                onChange={handleInputChange}
                                placeholder="e.g. Indoor, Outdoor, Water"
                                required
                                className="form-input"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label htmlFor="scoringMethod">Scoring Method</label>
                            <input
                                type="text"
                                id="scoringMethod"
                                name="scoringMethod"
                                value={formData.scoringMethod}
                                onChange={handleInputChange}
                                placeholder="e.g. Points-based, Goals-based, Time-based"
                                required
                                className="form-input"
                            />
                        </div>
                        
                        <button type="submit" className="submit-btn" disabled={loading}>
                            {loading ? (
                                <>
                                    <span className="loader"></span>
                                    <span>Processing...</span>
                                </>
                            ) : (
                                <>Get Recommendation</>
                            )}
                        </button>
                    </form>
                    
                    {error && (
                        <div className="error-message">
                            {error}
                        </div>
                    )}
                    
                    {result && (
                        <div className="result-container">
                            <h3>Predicted Sport</h3>
                            <div className="result">{result}</div>
    
                            {explanation && (
                                <div className="prescription">
                                    <h4>Prescription for the Sport:</h4>
                                    <div className="prescription-text">{explanation}</div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </main>
            
            <footer>
                <p>Rule-Based Classifier</p>
            </footer>
        </div>
    );
}

export default App;
