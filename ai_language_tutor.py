import streamlit as st
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import random
import json
from gtts import gTTS
from io import BytesIO
import base64

# Set page config first
st.set_page_config(
    page_title="AI Language Tutor", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class LanguageAITutor:
    def __init__(self, target_language: str = "Spanish"):
        self.target_language = target_language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Show device info
        st.sidebar.info(f"🚀 Using: {self.device.upper()}")
        
        # Initialize models with progress
        with st.spinner("Loading translation model..."):
            try:
                if target_language == "Spanish":
                    self.translator = pipeline(
                        "translation_en_to_es", 
                        device=0 if self.device == "cuda" else -1
                    )
                else:  # French
                    self.translator = pipeline(
                        "translation_en_to_fr",
                        device=0 if self.device == "cuda" else -1
                    )
            except Exception as e:
                st.error(f"Translation model loading failed: {e}")
                self.translator = None
        
        with st.spinner("Loading similarity model..."):
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                st.error(f"Similarity model loading failed: {e}")
                self.similarity_model = None
        
        # Vocabulary database
        self.vocabulary = self._initialize_vocabulary()
        
        # Learning progress
        self.user_progress = {
            "vocabulary_learned": set(),
            "lessons_completed": [],
            "current_level": "beginner"
        }
    
    def _initialize_vocabulary(self) -> dict:
        """Initialize vocabulary database"""
        if self.target_language == "Spanish":
            return {
                "beginner": [
                    {"english": "hello", "target": "hola", "category": "greetings"},
                    {"english": "goodbye", "target": "adiós", "category": "greetings"},
                    {"english": "please", "target": "por favor", "category": "courtesy"},
                    {"english": "thank you", "target": "gracias", "category": "courtesy"},
                    {"english": "yes", "target": "sí", "category": "basics"},
                    {"english": "no", "target": "no", "category": "basics"},
                    {"english": "water", "target": "agua", "category": "food"},
                    {"english": "food", "target": "comida", "category": "food"},
                    {"english": "house", "target": "casa", "category": "places"},
                    {"english": "car", "target": "coche", "category": "transportation"}
                ],
                "intermediate": [
                    {"english": "I would like to eat", "target": "me gustaría comer", "category": "phrases"},
                    {"english": "Where is the bathroom?", "target": "¿Dónde está el baño?", "category": "questions"},
                    {"english": "How much does it cost?", "target": "¿Cuánto cuesta?", "category": "questions"},
                    {"english": "My name is", "target": "Me llamo", "category": "introductions"},
                    {"english": "I don't understand", "target": "No entiendo", "category": "conversation"}
                ],
                "advanced": [
                    {"english": "I would like to make a reservation", "target": "Me gustaría hacer una reservación", "category": "travel"},
                    {"english": "Could you help me please?", "target": "¿Podría ayudarme por favor?", "category": "requests"},
                    {"english": "What do you recommend?", "target": "¿Qué recomienda?", "category": "questions"}
                ]
            }
        else:  # French
            return {
                "beginner": [
                    {"english": "hello", "target": "bonjour", "category": "greetings"},
                    {"english": "goodbye", "target": "au revoir", "category": "greetings"},
                    {"english": "please", "target": "s'il vous plaît", "category": "courtesy"},
                    {"english": "thank you", "target": "merci", "category": "courtesy"},
                    {"english": "yes", "target": "oui", "category": "basics"},
                    {"english": "no", "target": "non", "category": "basics"},
                    {"english": "water", "target": "eau", "category": "food"},
                    {"english": "food", "target": "nourriture", "category": "food"},
                    {"english": "house", "target": "maison", "category": "places"},
                    {"english": "car", "target": "voiture", "category": "transportation"}
                ],
                "intermediate": [
                    {"english": "I would like to eat", "target": "je voudrais manger", "category": "phrases"},
                    {"english": "Where is the bathroom?", "target": "Où sont les toilettes?", "category": "questions"},
                    {"english": "How much does it cost?", "target": "Combien ça coûte?", "category": "questions"},
                    {"english": "My name is", "target": "Je m'appelle", "category": "introductions"},
                    {"english": "I don't understand", "target": "Je ne comprends pas", "category": "conversation"}
                ],
                "advanced": [
                    {"english": "I would like to make a reservation", "target": "Je voudrais faire une réservation", "category": "travel"},
                    {"english": "Could you help me please?", "target": "Pourriez-vous m'aider s'il vous plaît?", "category": "requests"},
                    {"english": "What do you recommend?", "target": "Que recommandez-vous?", "category": "questions"}
                ]
            }
    
    def translate_text(self, text: str) -> str:
        """Translate text to target language"""
        if not self.translator:
            return "Translation service unavailable"
        
        try:
            result = self.translator(text)
            return result[0]['translation_text']
        except Exception as e:
            return f"Translation error: {e}"
    
    def check_answer_similarity(self, user_answer: str, correct_answer: str) -> float:
        """Check similarity between user answer and correct answer"""
        if not self.similarity_model:
            return 1.0 if user_answer.lower().strip() == correct_answer.lower().strip() else 0.0
        
        try:
            embeddings = self.similarity_model.encode([user_answer, correct_answer], convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
            return cosine_scores.item()
        except:
            return 1.0 if user_answer.lower().strip() == correct_answer.lower().strip() else 0.0
    
    def generate_audio(self, text: str) -> str:
        """Generate audio for text-to-speech"""
        try:
            lang_code = 'es' if self.target_language == "Spanish" else 'fr'
            tts = gTTS(text=text, lang=lang_code, slow=False)
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Encode audio for Streamlit
            audio_base64 = base64.b64encode(audio_buffer.read()).decode()
            audio_html = f'''
                <audio controls autoplay style="width: 100%">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            '''
            return audio_html
        except Exception as e:
            return f"<p style='color: red;'>Audio generation failed: {e}</p>"
    
    def get_vocabulary_lesson(self, level: str = "beginner") -> list:
        """Get vocabulary for a specific level"""
        return self.vocabulary.get(level, [])
    
    def create_interactive_exercise(self) -> dict:
        """Create an interactive exercise"""
        level = self.user_progress["current_level"]
        vocab_list = self.get_vocabulary_lesson(level)
        
        if not vocab_list:
            return {"type": "complete", "message": "🎉 Congratulations! You've completed all exercises for this level!"}
        
        exercise_type = random.choice(["translation", "matching", "fill_blank"])
        item = random.choice([v for v in vocab_list if v['english'] not in self.user_progress["vocabulary_learned"] or random.random() < 0.3])
        
        if exercise_type == "translation":
            return {
                "type": "translation",
                "question": f"Translate to {self.target_language}: '{item['english']}'",
                "correct_answer": item['target'],
                "hint": f"Category: {item['category']}",
                "vocab_item": item
            }
        
        elif exercise_type == "matching":
            options = [item['target']]
            # Add distractors
            other_items = [v for v in vocab_list if v != item]
            distractors = random.sample(other_items, min(3, len(other_items)))
            options.extend([d['target'] for d in distractors])
            random.shuffle(options)
            
            return {
                "type": "matching",
                "question": f"Match the English word/phrase: '{item['english']}'",
                "options": options,
                "correct_answer": item['target'],
                "vocab_item": item
            }
        
        else:  # fill_blank
            sentence_templates = [
                f"The {self.target_language} word for '{item['english']}' is: _____",
                f"I need to say '{item['english']}' in {self.target_language}. It is: _____",
                f"How do you say '{item['english']}' in {self.target_language}? _____"
            ]
            
            return {
                "type": "fill_blank",
                "question": random.choice(sentence_templates),
                "correct_answer": item['target'],
                "vocab_item": item
            }

def initialize_session_state():
    """Initialize session state variables"""
    if 'tutor' not in st.session_state:
        st.session_state.tutor = None
    if 'current_exercise' not in st.session_state:
        st.session_state.current_exercise = None
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'exercises_completed' not in st.session_state:
        st.session_state.exercises_completed = 0
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = ""

def main():
    st.title("🎓 AI Language Learning Tutor")
    st.markdown("### Learn languages interactively with AI-powered exercises!")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        target_language = st.selectbox(
            "Choose Language to Learn", 
            ["Spanish", "French"],
            key="language_select"
        )
        
        difficulty = st.selectbox(
            "Difficulty Level", 
            ["beginner", "intermediate", "advanced"],
            key="difficulty_select"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Initialize Tutor", use_container_width=True):
                with st.spinner("Loading AI models... This may take a minute."):
                    st.session_state.tutor = LanguageAITutor(target_language)
                    st.session_state.tutor.user_progress["current_level"] = difficulty
                    st.session_state.score = 0
                    st.session_state.exercises_completed = 0
                    st.session_state.current_exercise = None
                    st.session_state.show_answer = False
                    st.session_state.user_answer = ""
                    st.success("Tutor initialized successfully!")
        
        with col2:
            if st.button("🔄 Reset Progress", use_container_width=True):
                st.session_state.score = 0
                st.session_state.exercises_completed = 0
                st.session_state.current_exercise = None
                st.session_state.show_answer = False
                st.session_state.user_answer = ""
                st.success("Progress reset!")
        
        st.header("📊 Progress Tracking")
        st.metric("🏆 Score", st.session_state.score)
        st.metric("✅ Exercises Completed", st.session_state.exercises_completed)
        
        if st.session_state.tutor:
            level = st.session_state.tutor.user_progress["current_level"]
            vocab_learned = len(st.session_state.tutor.user_progress["vocabulary_learned"])
            total_vocab = len(st.session_state.tutor.get_vocabulary_lesson(level))
            st.metric("📚 Vocabulary Mastered", f"{vocab_learned}/{total_vocab}")
        
        st.header("💡 Tips")
        st.info(
            "• Practice 15-30 minutes daily\n"
            "• Repeat words aloud\n"
            "• Use audio for pronunciation\n"
            "• Don't fear mistakes!\n"
            "• Think in the target language"
        )
    
    # Main content area
    if st.session_state.tutor is None:
        st.warning("👋 Welcome! Please click 'Initialize Tutor' in the sidebar to begin your language learning journey!")
        
        # Quick demo
        st.subheader("🚀 Quick Demo")
        st.write("This AI Tutor will help you learn languages with:")
        st.write("• 🤖 AI-powered translations and exercises")
        st.write("• 🔊 Text-to-speech pronunciation")
        st.write("• 📊 Progress tracking")
        st.write("• 🎯 Interactive quizzes")
        st.write("• 📚 Organized vocabulary by level")
        
        return
    
    tutor = st.session_state.tutor
    
    # Main interface in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"🌍 Learning {target_language}")
        
        # Translation tool
        with st.expander("🔤 Translation Tool", expanded=True):
            text_to_translate = st.text_input(
                "Enter English text to translate:",
                placeholder="Type something in English...",
                key="translate_input"
            )
            if text_to_translate:
                with st.spinner("Translating..."):
                    translation = tutor.translate_text(text_to_translate)
                    st.success(f"**Translation:** {translation}")
                    
                    # Audio playback
                    st.write("**Pronunciation:**")
                    audio_html = tutor.generate_audio(translation)
                    st.markdown(audio_html, unsafe_allow_html=True)
        
        # Interactive exercises
        st.header("💪 Practice Exercises")
        
        # Exercise controls
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("🎲 New Exercise", use_container_width=True):
                st.session_state.current_exercise = tutor.create_interactive_exercise()
                st.session_state.show_answer = False
                st.session_state.user_answer = ""
                st.rerun()
        
        # Display current exercise
        if st.session_state.current_exercise:
            exercise = st.session_state.current_exercise
            
            if exercise["type"] == "complete":
                st.balloons()
                st.success(exercise["message"])
                st.info("🎉 Try a higher difficulty level or reset your progress to practice again!")
            else:
                st.subheader("Current Exercise")
                st.write(f"**{exercise['question']}**")
                
                if "hint" in exercise:
                    st.caption(f"💡 {exercise['hint']}")
                
                # User input based on exercise type
                if exercise["type"] == "matching":
                    user_answer = st.radio(
                        "Select your answer:",
                        exercise["options"],
                        key="matching_answer"
                    )
                else:
                    user_answer = st.text_input(
                        "Your answer:",
                        placeholder=f"Type the {target_language} translation...",
                        key="text_answer"
                    )
                
                # Check answer button
                if st.button("✅ Check Answer", type="primary", use_container_width=True):
                    if exercise["type"] == "matching":
                        is_correct = user_answer == exercise["correct_answer"]
                        similarity = 1.0 if is_correct else 0.0
                    else:
                        similarity = tutor.check_answer_similarity(
                            user_answer.lower().strip(), 
                            exercise["correct_answer"].lower().strip()
                        )
                        is_correct = similarity > 0.7
                    
                    if is_correct:
                        st.success("🎉 Correct! Well done!")
                        st.session_state.score += 10
                        st.session_state.exercises_completed += 1
                        
                        # Mark vocabulary as learned
                        if "vocab_item" in exercise:
                            tutor.user_progress["vocabulary_learned"].add(exercise["vocab_item"]['english'])
                        
                        # Show correct answer with audio
                        st.write(f"**Correct answer:** {exercise['correct_answer']}")
                        st.write("**Listen to pronunciation:**")
                        audio_html = tutor.generate_audio(exercise['correct_answer'])
                        st.markdown(audio_html, unsafe_allow_html=True)
                        
                        # Auto-generate new exercise after delay
                        st.write("Generating new exercise in 3 seconds...")
                        st.rerun()
                        
                    else:
                        st.error("❌ Not quite right. Try again!")
                        st.write(f"Similarity score: {similarity:.2%}")
                        
                        if similarity > 0.5:
                            st.info("🔍 You're close! Check spelling and try again.")
                        else:
                            st.info("💡 Listen to the pronunciation and try again!")
                
                # Show answer button
                if st.button("👀 Show Answer", use_container_width=True):
                    st.session_state.show_answer = True
                
                if st.session_state.show_answer:
                    st.info(f"**Answer:** {exercise['correct_answer']}")
                    st.write("**Pronunciation:**")
                    audio_html = tutor.generate_audio(exercise['correct_answer'])
                    st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.info("👆 Click 'New Exercise' to start practicing!")
    
    with col2:
        st.header("📚 Vocabulary Bank")
        
        level = tutor.user_progress["current_level"]
        vocab_list = tutor.get_vocabulary_lesson(level)
        
        st.write(f"**{level.title()} Level Vocabulary**")
        
        # Group vocabulary by category
        categories = {}
        for word in vocab_list:
            category = word['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(word)
        
        for category, words in categories.items():
            with st.expander(f"📂 {category.title()} ({len(words)} words)"):
                for word in words:
                    learned = "✅" if word['english'] in tutor.user_progress["vocabulary_learned"] else "📖"
                    st.write(f"{learned} **{word['english']}** → {word['target']}")
                    
                    # Mini audio player for each word
                    audio_html = tutor.generate_audio(word['target'])
                    st.markdown(audio_html, unsafe_allow_html=True)
                    st.markdown("---")
        
        # Quick stats
        st.header("🎯 Quick Stats")
        learned_count = len(tutor.user_progress["vocabulary_learned"])
        total_count = len(vocab_list)
        progress = learned_count / total_count if total_count > 0 else 0
        
        st.progress(progress)
        st.write(f"Vocabulary Progress: {learned_count}/{total_count} ({progress:.0%})")
        
        if progress > 0.8:
            st.success("🌟 Excellent progress! Consider moving to the next level.")
        elif progress > 0.5:
            st.info("📈 Good progress! Keep practicing.")

if __name__ == "__main__":
    main()