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
    const [confidence, setConfidence] = useState(null);
    const [alternatives, setAlternatives] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [typingEffect, setTypingEffect] = useState('');
    const [typingIndex, setTypingIndex] = useState(0);
    const [activeTab, setActiveTab] = useState('naiveBayes');

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
        setConfidence(null);
        setAlternatives([]);
        
        try {
            await new Promise(resolve => setTimeout(resolve, 800));
            
            const endpoint = activeTab === 'naiveBayes' 
                ? 'http://localhost:5000/api/sports' 
                : 'http://localhost:5000/api/prescriptive';
            
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
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to get recommendation');
            }

            // Handle both endpoints' response formats
            if (activeTab === 'naiveBayes') {
                setResult(data.recommended_sport);
            } else {
                setResult(data.recommendation);
            }
            
            setExplanation(data.explanation);
            setConfidence(data.confidence);
            setAlternatives(data.alternatives || []);
            
        } catch (err) {
            console.error('Error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleTabChange = (tab) => {
        setActiveTab(tab);
        setFormData({
            players: '',
            equipment: '',
            playingArea: '',
            scoringMethod: ''
        });
        setResult('');
        setError('');
        setExplanation('');
        setConfidence(null);
        setAlternatives([]);
        
        setTypingEffect('');
        setTypingIndex(0);
    };

    return (
        <div className="container">
            <nav className="navbar">
                <ul>
                    <li className={activeTab === 'naiveBayes' ? 'active' : ''} 
                        onClick={() => handleTabChange('naiveBayes')}>
                        ITBAN 3
                    </li>
                    <li className={activeTab === 'ifThen' ? 'active' : ''} 
                        onClick={() => handleTabChange('ifThen')}>
                        ITBAN 4
                    </li>
                </ul>
            </nav>
            
            <header>
                <h1>Sports <span className="highlight">Recommendation</span></h1>
            </header>
            
            <main>
                <div className="content">
                    <h2>{activeTab === 'naiveBayes' ? 'Find Your Sport' : 'Sport Recommendation with Rules'}</h2>
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
                            <h3>Primary Recommendation</h3>
                            <div className="result">{result}</div>
                            
                            {confidence !== null && (
                                <div className="confidence-bar">
                                    <div className="confidence-label">
                                        Match Confidence: {Math.round(confidence * 100)}%
                                    </div>
                                    <div className="confidence-track">
                                        <div 
                                            className="confidence-fill" 
                                            style={{width: `${confidence * 100}%`}}
                                        />
                                    </div>
                                </div>
                            )}

                            {explanation && (
                                <div className="explanation">
                                    <div className="explanation-tag">Why this sport?</div>
                                    <div className="explanation-text">{explanation}</div>
                                </div>
                            )}
                            
                            {alternatives && alternatives.length > 0 && (
                                <div className="alternatives">
                                    <div className="alternatives-tag">
                                        Alternative Recommendations
                                    </div>
                                    <div className="alternatives-list">
                                        {alternatives.map((alt, index) => (
                                            <div key={index} className="alternative-item">
                                                <div className="alternative-name">
                                                    {alt.sport}
                                                    <div className="alternative-confidence">
                                                        {Math.round(alt.confidence * 100)}% Match
                                                    </div>
                                                </div>
                                                <div className="alternative-description">
                                                    {alt.explanation}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </main>
            
            <footer>
                <p>{activeTab === 'naiveBayes' ? 'ITBAN 3 Naive Bayes' : 'ITBAN 4 Prescriptive Analytics'}</p>
            </footer>
        </div>
    );
}

export default App;
