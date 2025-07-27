"""
Interactive web demo for fake news detection.
Provides a user-friendly interface to test the models.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.pipeline import FakeNewsDetector, PipelineConfig
    from src.interpretability.explainer import ExplainerFactory, MultiExplainer
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection Demo",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .fake-prediction {
        background-color: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    
    .real-prediction {
        background-color: #e8f5e8;
        border-color: #4caf50;
        color: #2e7d32;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        color: #212529;
    }
    
    .metric-card h4 {
        color: #007bff;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #495057;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_detector():
    """Load the fake news detector with caching."""
    with st.spinner("Initializing models..."):
        try:
            # Try to load from config
            config_path = "configs/default_config.yaml"
            if Path(config_path).exists():
                detector = FakeNewsDetector.from_config_file(config_path)
            else:
                detector = FakeNewsDetector()
            
            # Load datasets
            datasets = detector.load_datasets()
            
            # Train traditional models if datasets available
            if datasets:
                detector.train_traditional_models()
                detector.create_ensemble()
                return detector, True
            else:
                st.warning("No datasets found. Using demo mode with synthetic data.")
                return detector, False
                
        except Exception as e:
            st.error(f"Failed to initialize detector: {e}")
            return None, False

def create_sample_texts():
    """Create sample texts for testing."""
    return {
        "Fake News Examples": [
            """BREAKING: UFO crash site discovered, government cover-up exposed!
            
            Exclusive leaked footage shows alien spacecraft wreckage being secretly removed by military forces in Nevada desert. Multiple eyewitnesses report seeing strange lights and hearing unexplained sounds before government agents arrived to secure the area. Sources within the military confirm that extraterrestrial technology has been recovered and is being reverse-engineered in underground facilities.
            
            This proves what conspiracy theorists have been saying for decades - aliens are real and the government knows it! The cover-up goes all the way to the top. They don't want you to know the truth about what's really happening. Secret documents obtained by our investigative team reveal that this is just one of many crash sites that have been hidden from public knowledge.
            
            The military-industrial complex has been working with these alien beings for decades, trading human technology for advanced propulsion systems. Wake up, America! The truth is out there, and they're desperately trying to keep it hidden. Share this story before they delete it from the internet!""",
            
            """URGENT WARNING: 5G towers are killing everyone - scientists demand immediate shutdown!
            
            Secret government study reveals that 5G radiation is causing mass bird deaths and human illnesses worldwide. Leaked documents from telecommunications companies show they knew about the deadly effects but chose to hide them from the public for profit. Children are especially vulnerable to the harmful electromagnetic frequencies that are literally cooking our brains.
            
            Thousands of scientists worldwide are calling for an immediate ban on 5G technology before it's too late. The wireless industry is spending millions to suppress this information and silence whistleblowers. Don't believe the mainstream media lies - they're being paid off by Big Tech to keep you in the dark about this massive conspiracy.
            
            Symptoms include headaches, fatigue, and decreased cognitive function. Protect your family by moving away from 5G towers immediately. The government won't tell you this, but aluminum foil can block the harmful radiation. Share this urgent warning with everyone you know!""",
            
            """EXPOSED: The real reason they don't want you to know about this miracle cure!
            
            Big Pharma doesn't want you to discover this one simple trick that eliminates ALL diseases using common household items! This revolutionary treatment has been suppressed by the medical establishment for decades because it threatens their trillion-dollar industry. Doctors are furious because this natural remedy makes their expensive treatments obsolete.
            
            Local mom discovers ancient secret that cures cancer, diabetes, and heart disease in just 24 hours! The pharmaceutical companies are desperately trying to shut down this information before it goes viral. They've already bought off the mainstream media to prevent coverage of this groundbreaking discovery.
            
            Don't let them silence the truth! This simple method costs less than $5 and uses ingredients you already have at home. Thousands of people have already been cured, but the government is trying to make it illegal. Act now before they ban this life-saving information forever!"""
        ],
        "Real News Examples": [
            """President Biden signs infrastructure bill into law
            
            President Joe Biden signed a $1.2 trillion infrastructure bill into law on Monday, fulfilling a key campaign promise to rebuild America's roads, bridges, broadband networks and other critical infrastructure. The bipartisan legislation passed the House on Friday after months of negotiations between Democrats and Republicans.
            
            The bill includes $550 billion in new federal spending over five years on transportation, utilities and broadband. It allocates $110 billion for roads and bridges, $66 billion for passenger and freight rail, and $65 billion for broadband expansion. The legislation also includes $55 billion for water infrastructure improvements.
            
            "This is a once-in-a-generation investment in our nation's infrastructure," Biden said during the signing ceremony at the White House. The president was joined by lawmakers from both parties who supported the legislation. The Congressional Budget Office estimates the bill will add $256 billion to federal deficits over the next decade.
            
            Implementation of the infrastructure projects will begin immediately, with the Department of Transportation expected to announce the first round of funding allocations within 90 days. State and local governments have already begun submitting project proposals for federal review and approval.""",
            
            """Stock market closes higher as investors await Federal Reserve meeting
            
            U.S. stocks finished higher on Tuesday as investors awaited the Federal Reserve's policy meeting this week. The Dow Jones Industrial Average rose 104 points to close at 35,462. The S&P 500 gained 0.3% to 4,630, while the Nasdaq Composite advanced 0.6% to 15,331.
            
            Technology stocks led the gains, with shares of Microsoft and Apple both rising more than 1%. Energy and financial sectors also posted solid gains, while consumer staples and utilities lagged. Trading volume was lighter than average as investors positioned ahead of the Fed's two-day meeting.
            
            Federal Reserve officials are expected to announce another interest rate increase as they continue efforts to combat inflation. Most economists anticipate a 0.75 percentage point increase, which would bring the federal funds rate to its highest level in 15 years.
            
            Bond yields moved higher in anticipation of the Fed's decision, with the 10-year Treasury yield rising to 4.1%. The dollar strengthened against major currencies as investors priced in continued monetary policy tightening.""",
            
            """NASA successfully launches James Webb Space Telescope
            
            NASA successfully launched the James Webb Space Telescope on Saturday morning from French Guiana, beginning a million-mile journey to its destination in space. The $10 billion observatory, considered the successor to the Hubble Space Telescope, lifted off aboard an Ariane 5 rocket at 7:20 a.m. EST.
            
            The telescope is expected to revolutionize astronomy by peering deeper into space and further back in time than ever before. It will study the formation of the first stars and galaxies, examine the atmospheres of exoplanets, and investigate the origins of the universe.
            
            The Webb telescope features a primary mirror that is nearly three times larger than Hubble's, allowing it to collect more light and see fainter objects. It will operate at extremely cold temperatures and observe primarily in infrared light, enabling it to see through cosmic dust that obscures visible light.
            
            The telescope will spend the next month traveling to its destination at the second Lagrange point, where it will undergo six months of commissioning before beginning scientific observations. NASA Administrator Bill Nelson called the launch "a tremendous achievement for humanity."""
        ]
    }

def predict_text(detector, text, model_type):
    """Make prediction and return results."""
    try:
        start_time = time.time()
        results = detector.predict([text], model_type=model_type)
        processing_time = time.time() - start_time
        
        prediction = results['labels'][0]
        probabilities = results['probabilities'][0]
        confidence = results['confidence'][0]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability_fake': probabilities[0],
            'probability_real': probabilities[1],
            'processing_time': processing_time,
            'model_used': model_type
        }
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def create_confidence_chart(probability_fake, probability_real):
    """Create a confidence visualization chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Fake', 'Real'],
            y=[probability_fake, probability_real],
            marker_color=['#ff4444', '#44ff44'],
            text=[f'{probability_fake:.1%}', f'{probability_real:.1%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Category",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def create_history_chart(history):
    """Create a chart showing prediction history."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Create timeline chart
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='confidence',
        color='prediction',
        size='confidence',
        hover_data=['model_used'],
        title="Prediction History",
        color_discrete_map={'Fake': '#ff4444', 'Real': '#44ff44'}
    )
    
    fig.update_layout(height=400)
    return fig

def explain_prediction(detector, text, model_type):
    """Generate explanation for the prediction."""
    try:
        # Get the specific model
        if model_type == 'ensemble' and hasattr(detector, 'ensemble_model'):
            model = detector.ensemble_model
            # Get ensemble-specific explanations
            explanation = get_ensemble_explanation(detector, text)
            return explanation
        elif model_type in detector.traditional_models:
            model = detector.traditional_models[model_type]
        else:
            return {"error": "Model not available for explanation"}
        
        # Create multi-explainer for comprehensive analysis
        explainer = MultiExplainer(model, model_type)
        explanation = explainer.explain_prediction(text)
        
        # If multi-explainer fails, fall back to simple feature importance
        if 'error' in explanation or not explanation.get('consolidated_features'):
            simple_explainer = ExplainerFactory.create_explainer(
                'feature_importance', model, model_type
            )
            explanation = simple_explainer.explain_prediction(text)
        
        return explanation
        
    except Exception as e:
        return {"error": f"Explanation failed: {e}"}

def get_ensemble_explanation(detector, text):
    """Generate ensemble-specific explanation with model contributions."""
    try:
        ensemble_model = detector.ensemble_model
        
        # Get individual model predictions
        model_contributions = ensemble_model.get_model_contributions([text])
        
        # Get ensemble prediction
        ensemble_prediction = ensemble_model.predict([text])[0]
        ensemble_proba = ensemble_model.predict_proba([text])[0]
        
        # Analyze model agreement
        individual_predictions = {}
        individual_confidences = {}
        
        for model_name, proba in model_contributions.items():
            pred = np.argmax(proba[0])
            conf = np.max(proba[0])
            individual_predictions[model_name] = 'Real' if pred == 1 else 'Fake'
            individual_confidences[model_name] = {
                'prediction': 'Real' if pred == 1 else 'Fake',
                'confidence': float(conf),
                'probabilities': {
                    'Fake': float(proba[0][0]),
                    'Real': float(proba[0][1])
                }
            }
        
        # Calculate agreement metrics
        predictions_list = [individual_predictions[name] for name in individual_predictions]
        agreement_score = predictions_list.count(max(set(predictions_list), key=predictions_list.count)) / len(predictions_list)
        
        # Determine ensemble method info
        method_info = {
            'method': ensemble_model.combination_method,
            'weights': dict(zip(ensemble_model.model_names, ensemble_model.weights)) if hasattr(ensemble_model, 'weights') and ensemble_model.weights else None
        }
        
        return {
            'method': 'ensemble',
            'ensemble_prediction': 'Real' if ensemble_prediction == 1 else 'Fake',
            'ensemble_confidence': float(max(ensemble_proba)),
            'ensemble_probabilities': {
                'Fake': float(ensemble_proba[0]),
                'Real': float(ensemble_proba[1])
            },
            'individual_models': individual_confidences,
            'agreement_score': agreement_score,
            'method_info': method_info,
            'explanation_text': generate_ensemble_explanation_text(
                individual_predictions, agreement_score, method_info
            )
        }
        
    except Exception as e:
        return {"error": f"Ensemble explanation failed: {e}"}

def generate_ensemble_explanation_text(individual_predictions, agreement_score, method_info):
    """Generate human-readable explanation for ensemble predictions."""
    explanation_parts = []
    
    # Agreement analysis
    if agreement_score == 1.0:
        explanation_parts.append("All models agree on this prediction.")
    elif agreement_score >= 0.67:
        explanation_parts.append("Most models agree on this prediction.")
    else:
        explanation_parts.append("Models disagree on this prediction.")
    
    # Method explanation
    method = method_info['method']
    if method == 'weighted_voting':
        explanation_parts.append("The ensemble uses weighted voting, where models with better performance have more influence.")
    elif method == 'stacking':
        explanation_parts.append("The ensemble uses stacking, where a meta-model learns to combine the predictions.")
    elif method == 'confidence_based':
        explanation_parts.append("The ensemble weights predictions based on individual model confidence.")
    
    # Individual model breakdown
    fake_count = sum(1 for pred in individual_predictions.values() if pred == 'Fake')
    real_count = len(individual_predictions) - fake_count
    explanation_parts.append(f"Individual models: {fake_count} predicted Fake, {real_count} predicted Real.")
    
    return " ".join(explanation_parts)

def display_ensemble_explanation(explanation):
    """Display ensemble-specific explanation with visualizations."""
    # Model agreement section
    st.subheader("Model Agreement Analysis")
    agreement_score = explanation.get('agreement_score', 0)
    
    # Create agreement visualization
    col1, col2 = st.columns([1, 2])
    with col1:
        # Agreement score gauge
        agreement_color = "green" if agreement_score >= 0.8 else "orange" if agreement_score >= 0.6 else "red"
        st.metric("Agreement Score", f"{agreement_score:.1%}")
        
        if agreement_score == 1.0:
            st.success("🎯 Perfect Agreement")
        elif agreement_score >= 0.8:
            st.info("✅ Strong Agreement")
        elif agreement_score >= 0.6:
            st.warning("⚠️ Moderate Agreement")
        else:
            st.error("❌ Low Agreement")
    
    with col2:
        # Individual model predictions
        individual_models = explanation.get('individual_models', {})
        if individual_models:
            st.write("**Individual Model Predictions:**")
            for model_name, model_data in individual_models.items():
                prediction = model_data['prediction']
                confidence = model_data['confidence']
                icon = "✅" if prediction == "Real" else "❌"
                st.write(f"{icon} **{model_name.upper()}**: {prediction} ({confidence:.1%} confidence)")
    
    # Ensemble method explanation
    st.subheader("Ensemble Method Details")
    method_info = explanation.get('method_info', {})
    method = method_info.get('method', 'unknown')
    
    if method == 'weighted_voting':
        st.info("**Weighted Voting**: Each model's prediction is weighted by its performance on validation data.")
        weights = method_info.get('weights')
        if weights:
            st.write("**Model Weights:**")
            weight_df = pd.DataFrame(list(weights.items()), columns=['Model', 'Weight'])
            weight_df['Weight %'] = (weight_df['Weight'] * 100).round(1)
            st.dataframe(weight_df, use_container_width=True)
            
    elif method == 'stacking':
        st.info("**Stacking**: A meta-model learns how to best combine the predictions from individual models.")
        
    elif method == 'confidence_based':
        st.info("**Confidence-Based**: Predictions are weighted based on each model's confidence for this specific input.")
    
    # Individual model confidence breakdown chart
    if individual_models:
        st.subheader("Individual Model Confidence Breakdown")
        create_model_confidence_chart(individual_models)

def create_model_confidence_chart(individual_models):
    """Create a chart showing individual model confidences."""
    models = []
    fake_probs = []
    real_probs = []
    
    for model_name, model_data in individual_models.items():
        models.append(model_name.upper())
        fake_probs.append(model_data['probabilities']['Fake'])
        real_probs.append(model_data['probabilities']['Real'])
    
    # Create stacked bar chart
    fig = go.Figure(data=[
        go.Bar(name='Fake', x=models, y=fake_probs, marker_color='#ff4444'),
        go.Bar(name='Real', x=models, y=real_probs, marker_color='#44ff44')
    ])
    
    fig.update_layout(
        barmode='stack',
        title="Individual Model Confidence Distribution",
        xaxis_title="Models",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    st.markdown('<h1 class="main-header">Fake News Detection Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Load detector
    if st.session_state.detector is None:
        detector, models_loaded = load_detector()
        st.session_state.detector = detector
        st.session_state.models_loaded = models_loaded
    
    detector = st.session_state.detector
    
    if detector is None:
        st.error("Failed to load detector. Please check your setup.")
        return
    
    # Model selection
    available_models = ['tfidf']
    if hasattr(detector, 'transformer_model') and detector.transformer_model is not None:
        available_models.append('deberta')
    if hasattr(detector, 'ensemble_model') and detector.ensemble_model is not None:
        available_models.append('ensemble')
    
    model_type = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=len(available_models)-1 if 'ensemble' in available_models else 0
    )
    
    # Model info
    st.sidebar.markdown("### Model Information")
    if model_type == 'tfidf':
        st.sidebar.info("**TF-IDF**: Traditional model using term frequency-inverse document frequency")
    elif model_type == 'deberta':
        st.sidebar.info("**DeBERTa-v3**: Advanced transformer model with disentangled attention")
    elif model_type == 'ensemble':
        st.sidebar.info("**Ensemble**: Combination of TF-IDF + DeBERTa transformer for optimal accuracy")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Text Analysis")
        
        # Text input
        input_method = st.radio(
            "Choose input method:",
            ["Type your own text", "Use sample texts"]
        )
        
        if input_method == "Type your own text":
            text_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Paste or type the news article you want to analyze..."
            )
        else:
            sample_texts = create_sample_texts()
            category = st.selectbox("Select category:", list(sample_texts.keys()))
            text_choice = st.selectbox("Select example:", sample_texts[category])
            text_input = text_choice
            
            # Show the selected text
            st.text_area("Selected text:", value=text_input, height=100, disabled=True)
        
        # Prediction button
        if st.button("Analyze Text", type="primary", disabled=not text_input):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predict_text(detector, text_input.strip(), model_type)
                    
                    if result:
                        # Store in history
                        st.session_state.prediction_history.append({
                            'timestamp': pd.Timestamp.now(),
                            'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'model_used': result['model_used']
                        })
                        
                        # Display results
                        prediction_class = "fake-prediction" if result['prediction'] == 'Fake' else "real-prediction"
                        
                        st.markdown(f"""
                        <div class="prediction-box {prediction_class}">
                            <h3>Prediction: {result['prediction']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                            <p><strong>Model:</strong> {result['model_used'].upper()}</p>
                            <p><strong>Processing Time:</strong> {result['processing_time']*1000:.1f}ms</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Explanation section (moved above confidence chart)
                        st.subheader("Explanation")
                        with st.expander("Show detailed explanation", expanded=True):
                            explanation = explain_prediction(detector, text_input.strip(), model_type)
                            
                            if 'error' in explanation:
                                st.warning("Advanced explanation not available for this model. Using basic analysis.")
                                # Basic explanation based on prediction confidence
                                if result['confidence'] > 0.8:
                                    confidence_text = "very confident"
                                elif result['confidence'] > 0.6:
                                    confidence_text = "moderately confident"
                                else:
                                    confidence_text = "somewhat uncertain"
                                
                                basic_explanation = f"The model is {confidence_text} that this text is {result['prediction'].lower()}."
                                
                                if result['prediction'] == 'Fake':
                                    basic_explanation += " This prediction may be based on sensational language, emotional appeals, lack of credible sources, or other patterns commonly found in misinformation."
                                else:
                                    basic_explanation += " This prediction may be based on factual reporting style, named sources, balanced perspective, or other patterns typical of legitimate news."
                                
                                st.write(basic_explanation)
                            else:
                                # Enhanced explanation with interpretability
                                explanation_text = explanation.get('explanation_text', 'Analysis based on learned patterns.')
                                st.write(explanation_text)
                                
                                # Ensemble-specific explanations
                                if explanation.get('method') == 'ensemble':
                                    display_ensemble_explanation(explanation)
                                else:
                                    # Show key words found in text
                                    if 'text_features' in explanation and explanation['text_features']:
                                        st.subheader("Key Words Influencing Prediction:")
                                        features_df = pd.DataFrame(explanation['text_features'])
                                        if not features_df.empty:
                                            # Display as columns for better readability
                                            feature_names = features_df['feature'].tolist()[:8]
                                            if feature_names:
                                                cols = st.columns(min(4, len(feature_names)))
                                                for i, feature in enumerate(feature_names):
                                                    with cols[i % len(cols)]:
                                                        st.write(f"• **{feature}**")
                                    
                                    # Global model patterns
                                    if 'global_importance' in explanation:
                                        st.subheader("Common Patterns in Training Data:")
                                        col_fake, col_real = st.columns(2)
                                        with col_fake:
                                            st.write("**Typical Fake News Indicators:**")
                                            fake_indicators = explanation['global_importance'].get('fake_indicators', [])
                                            for indicator in fake_indicators[:5]:
                                                st.write(f"• {indicator}")
                                        
                                        with col_real:
                                            st.write("**Typical Real News Indicators:**")
                                            real_indicators = explanation['global_importance'].get('real_indicators', [])
                                            for indicator in real_indicators[:5]:
                                                st.write(f"• {indicator}")
                        
                        # Confidence visualization (moved after explanation)
                        st.subheader("Prediction Confidence")
                        st.plotly_chart(
                            create_confidence_chart(
                                result['probability_fake'], 
                                result['probability_real']
                            ),
                            use_container_width=True
                        )
            else:
                st.warning("Please enter some text to analyze.")
    
    with col2:
        st.header("Statistics")
        
        # Model performance metrics (from README)
        st.markdown("""
        <div class="metric-card">
            <h4>Model Performance (ISOT Dataset)</h4>
            <p><strong>TF-IDF F1 Score:</strong> 93.6%</p>
            <p><strong>DeBERTa F1 Score:</strong> 95.6%</p>
            <p><strong>Ensemble F1 Score:</strong> 96.6%</p>
            <p><em>Cross-domain F1: ~53-63%</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
            
            # Summary stats
            recent_predictions = st.session_state.prediction_history[-10:]
            fake_count = sum(1 for p in recent_predictions if p['prediction'] == 'Fake')
            real_count = len(recent_predictions) - fake_count
            
            col_fake, col_real = st.columns(2)
            with col_fake:
                st.metric("Fake", fake_count)
            with col_real:
                st.metric("Real", real_count)
            
            # History chart
            if len(st.session_state.prediction_history) > 1:
                history_chart = create_history_chart(st.session_state.prediction_history)
                if history_chart:
                    st.plotly_chart(history_chart, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Tips section
        st.subheader("Tips")
        st.markdown("""
        **Fake news often contains:**
        - Sensational language (SHOCKING, AMAZING)
        - Emotional appeals
        - Lack of credible sources
        - Grammatical errors
        - Clickbait headlines
        
        **Real news typically has:**
        - Factual reporting
        - Named sources
        - Balanced perspective
        - Professional writing
        - Verifiable information
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with Streamlit | Fake News Detection ML Pipeline</p>
        <p>Warning: This is a demo system. Always verify information from multiple sources.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()