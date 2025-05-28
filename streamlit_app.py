import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer # Not currently used for direct similarity, but good to keep if you plan TF-IDF analysis
from collections import Counter
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch # For better positioning
import base64
import time

# Set page configuration first
st.set_page_config(
    page_title="Pro Resume-JD Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NLTK Data Download ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("Downloading NLTK data..."):
            nltk.download('punkt')
            nltk.download('stopwords')

download_nltk_data()

# --- Helper Functions ---
def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if not text.strip():
            st.error("No text could be extracted from the PDF. It may be image-based or empty.")
            return ""
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def preprocess_text(text):
    """Cleans and tokenizes text."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens), tokens

@st.cache_data(show_spinner="Fetching embeddings from Gemini API...")
def get_gemini_embeddings(text):
    """Fetches text embeddings using the Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Gemini API key not found in environment variables.")
        return None
    try:
        genai.configure(api_key=api_key)
        response = genai.embed_content(
            model="models/text-embedding-004", # A good general-purpose embedding model
            content=text
        )
        if response and 'embedding' in response:
            return np.array(response['embedding'])
        else:
            st.error("No embedding found in response. Check model capabilities or API key.")
            return None
    except Exception as e:
        st.error(f"Error fetching embeddings: {e}. Ensure the API key is correct and you have network access.")
        return None

def compute_similarity(resume_text, jd_text):
    """Computes cosine similarity between two texts."""
    resume_embedding = get_gemini_embeddings(resume_text)
    jd_embedding = get_gemini_embeddings(jd_text)
    if resume_embedding is not None and jd_embedding is not None:
        return cosine_similarity([resume_embedding], [jd_embedding])[0][0]
    return 0

def get_keyword_frequencies(tokens):
    """Returns top 10 most frequent keywords."""
    return Counter(tokens).most_common(10)

def get_keyword_overlap(resume_tokens, jd_tokens):
    """Computes overlapping keywords and their frequencies."""
    resume_freq = Counter(resume_tokens)
    jd_freq = Counter(jd_tokens)
    common_keywords = set(resume_freq.keys()) & set(jd_freq.keys())

    # Sort common keywords by their combined frequency for better visualization
    sorted_common = sorted(common_keywords, key=lambda x: resume_freq.get(x,0) + jd_freq.get(x,0), reverse=True)

    return {kw: {'resume': resume_freq.get(kw, 0), 'jd': jd_freq.get(kw, 0)} for kw in sorted_common}

def generate_wordcloud_image(text, title):
    """Generates a word cloud from text and returns it as a PIL Image."""
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig) # Close the plot to free memory
    img_buffer.seek(0)
    return Image.open(img_buffer)

def generate_pdf_report(similarity_score, resume_keywords, jd_keywords, overlap_data):
    """Generates a downloadable PDF report."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, height - inch, "Resume-JD Match Report")
    c.setFont("Helvetica", 10)
    c.drawString(inch, height - inch - 20, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(inch, height - inch - 30, width - inch, height - inch - 30)

    # Match Score
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, height - inch - 60, "Overall Match Score:")
    c.setFont("Helvetica", 12)
    c.drawString(inch, height - inch - 80, f"{similarity_score*100:.1f}%")

    # Top Resume Keywords
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, height - inch - 120, "Top 10 Resume Keywords:")
    y = height - inch - 140
    for i, (kw, freq) in enumerate(resume_keywords):
        if y < inch: # New page if content goes beyond current page
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(inch, height - inch, "Top 10 Resume Keywords (cont.):")
            y = height - inch - 20
        c.drawString(inch + 20, y, f"- {kw} ({freq})")
        y -= 15

    # Top Job Description Keywords
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y - 30, "Top 10 Job Description Keywords:")
    y -= 50
    for i, (kw, freq) in enumerate(jd_keywords):
        if y < inch:
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(inch, height - inch, "Top 10 Job Description Keywords (cont.):")
            y = height - inch - 20
        c.drawString(inch + 20, y, f"- {kw} ({freq})")
        y -= 15

    # Keyword Overlap
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y - 30, "Top 10 Overlapping Keywords:")
    y -= 50
    overlap_df = pd.DataFrame.from_dict(overlap_data, orient='index').reset_index()
    overlap_df.columns = ['Keyword', 'Resume Count', 'JD Count']
    overlap_df = overlap_df.head(10) # Limit to top 10

    for idx, row in overlap_df.iterrows():
        if y < inch:
            c.showPage()
            c.setFont("Helvetica-Bold", 12)
            c.drawString(inch, height - inch, "Top 10 Overlapping Keywords (cont.):")
            y = height - inch - 20
        c.drawString(inch + 20, y, f"- {row['Keyword']}: Resume={row['Resume Count']}, JD={row['JD Count']}")
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer

def create_overlap_bar_chart(overlap_data):
    """Generates a bar chart for keyword overlap."""
    if not overlap_data:
        return None

    # Limit to top 10 for readability
    keywords_to_plot = list(overlap_data.keys())[:10]
    resume_counts = [overlap_data[kw]['resume'] for kw in keywords_to_plot]
    jd_counts = [overlap_data[kw]['jd'] for kw in keywords_to_plot]

    df_plot = pd.DataFrame({
        'Keyword': keywords_to_plot,
        'Resume Frequency': resume_counts,
        'Job Description Frequency': jd_counts
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.set_index('Keyword').plot(kind='barh', ax=ax, cmap='Paired')
    ax.set_title('Top 10 Keyword Overlap Between Resume and Job Description')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Keywords')
    ax.invert_yaxis() # Highest frequency at the top
    plt.tight_layout()
    return fig

def create_radar_chart(categories, resume_scores, jd_scores):
    """Generates a radar chart for skill category comparison."""
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    # Ensure scores are a list and add the first element to the end to close the loop
    resume_scores_closed = list(resume_scores) + [resume_scores[0]]
    jd_scores_closed = list(jd_scores) + [jd_scores[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, resume_scores_closed, linewidth=2, linestyle='solid', label='Your Resume', color='blue', alpha=0.7)
    ax.fill(angles, resume_scores_closed, 'blue', alpha=0.1)

    ax.plot(angles, jd_scores_closed, linewidth=2, linestyle='solid', label='Job Description', color='red', alpha=0.7)
    ax.fill(angles, jd_scores_closed, 'red', alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1) # Scores are between 0 and 1
    ax.set_title("Skill Category Comparison", va='bottom', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    return fig

# --- Streamlit App ---

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 8px;
        padding: 10px 20px;
        color: #262730; /* Darker text for tabs */
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E5E7EB;
    }
    .stTabs [aria-selected="true"] {
        background-color: #D3F2D4 !important; /* Lighter green for active tab */
        border-bottom-color: #4CAF50 !important; /* Green underline for active tab */
    }
    .metric-card {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 3em;
        font-weight: bold;
        color: #262730;
    }
    .metric-label {
        font-size: 1.2em;
        color: #555;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Pro Resume-JD Analyzer")
st.info("Upload your resume (PDF), enter a job description, and provide your Google Gemini API key to get a detailed compatibility analysis.")

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("🔑 API Configuration")
    gemini_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your API key from [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)"
    )
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        st.success("API Key set!")
    else:
        st.warning("Please enter your Google Gemini API key to proceed.")

# --- Initialize session state for results ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {
        'similarity_score': 0,
        'resume_processed': "",
        'jd_processed': "",
        'resume_tokens': [],
        'jd_tokens': [],
        'resume_keywords': [],
        'jd_keywords': [],
        'overlap_data': {},
        'has_run_analysis': False # Flag to indicate if analysis has been performed
    }

# --- Tabs for Organization ---
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "🎯 Results", "📈 Visualizations", "💡 Suggestions"])

with tab1:
    st.subheader("Upload Resume & Enter Job Description")
    col_upload, col_jd = st.columns(2)
    with col_upload:
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", key="resume_uploader")
    with col_jd:
        jd_text = st.text_area("Paste Job Description", height=300, key="jd_input")

    similarity_threshold = st.slider(
        "Set Similarity Threshold for 'Good Match' (%)",
        min_value=0, max_value=100, value=70, step=5,
        help="Adjust the minimum percentage score considered a 'good match'."
    )
    st.session_state.similarity_threshold = similarity_threshold # Store in session state

    if uploaded_file and jd_text and os.getenv("GEMINI_API_KEY"):
        if st.button("Analyze Compatibility"):
            with st.spinner("Analyzing..."):
                progress = st.progress(0, text="Starting analysis...")

                # Step 1: Extract and preprocess texts
                progress.progress(10, text="Extracting text from resume...")
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text:
                    st.session_state.analysis_results['has_run_analysis'] = False
                    progress.empty() # Remove progress bar
                    st.stop() # Stop execution if text extraction fails

                progress.progress(30, text="Preprocessing texts...")
                resume_processed, resume_tokens = preprocess_text(resume_text)
                jd_processed, jd_tokens = preprocess_text(jd_text)

                # Step 2: Compute similarity
                progress.progress(50, text="Computing similarity score...")
                similarity_score = compute_similarity(resume_processed, jd_processed)
                if similarity_score is None: # Handle cases where embeddings failed
                    st.session_state.analysis_results['has_run_analysis'] = False
                    progress.empty()
                    st.stop()

                # Step 3: Keyword analysis
                progress.progress(80, text="Performing keyword analysis...")
                resume_keywords = get_keyword_frequencies(resume_tokens)
                jd_keywords = get_keyword_frequencies(jd_tokens)
                overlap_data = get_keyword_overlap(resume_tokens, jd_tokens)

                # Store results in session state
                st.session_state.analysis_results = {
                    'similarity_score': similarity_score,
                    'resume_processed': resume_processed,
                    'jd_processed': jd_processed,
                    'resume_tokens': resume_tokens,
                    'jd_tokens': jd_tokens,
                    'resume_keywords': resume_keywords,
                    'jd_keywords': jd_keywords,
                    'overlap_data': overlap_data,
                    'has_run_analysis': True
                }
                progress.progress(100, text="Analysis complete!")
                st.toast("Analysis complete! Check the 'Results', 'Visualizations', and 'Suggestions' tabs.", icon="✅")
                time.sleep(0.5) # Short delay for toast to be seen
                progress.empty() # Remove progress bar

                # Automatically switch to Results tab after analysis
                st.query_params["tab"] = "results" # This might require a full rerun for tabs to update in some Streamlit versions

    elif not (uploaded_file and jd_text and os.getenv("GEMINI_API_KEY")):
        st.warning("Please upload a resume, paste the job description, and ensure your Gemini API key is set to run the analysis.")

# --- Display Results Tab ---
with tab2:
    if st.session_state.analysis_results['has_run_analysis']:
        similarity_score = st.session_state.analysis_results['similarity_score']
        resume_keywords = st.session_state.analysis_results['resume_keywords']
        jd_keywords = st.session_state.analysis_results['jd_keywords']
        overlap_data = st.session_state.analysis_results['overlap_data']
        similarity_threshold = st.session_state.similarity_threshold

        st.header("🎯 Match Score")
        score_percentage = similarity_score * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Compatibility Score</div>
                <div class="metric-value">{score_percentage:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if score_percentage >= similarity_threshold:
            st.success("🎉 Excellent Match! Your resume highly aligns with the job description.")
        elif score_percentage >= (similarity_threshold - 30): # Example threshold for 'good'
            st.warning("👍 Good Match. You have a decent alignment, but there's room for improvement.")
        else:
            st.error("Needs Improvement. Your resume has low similarity. Consider tailoring it further.")

        st.markdown("---")
        st.header("📋 Key Term Frequencies")

        col_res_kws, col_jd_kws = st.columns(2)
        with col_res_kws:
            st.subheader("Top Resume Keywords")
            df_resume_kws = pd.DataFrame(resume_keywords, columns=['Keyword', 'Frequency'])
            st.dataframe(df_resume_kws, hide_index=True, use_container_width=True)

        with col_jd_kws:
            st.subheader("Top Job Description Keywords")
            df_jd_kws = pd.DataFrame(jd_keywords, columns=['Keyword', 'Frequency'])
            st.dataframe(df_jd_kws, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.header("🔗 Overlapping Keywords")
        st.write("Keywords present in both your resume and the job description, with their respective frequencies.")
        if overlap_data:
            overlap_df = pd.DataFrame.from_dict(overlap_data, orient='index').reset_index()
            overlap_df.columns = ['Keyword', 'Resume Count', 'JD Count']
            overlap_df = overlap_df.sort_values(by=['Resume Count', 'JD Count'], ascending=False).head(20) # Show top 20 by combined count
            st.dataframe(overlap_df, hide_index=True, use_container_width=True)
        else:
            st.info("No common keywords found. Try adjusting your resume or the job description.")

        # Downloadable report
        pdf_buffer = generate_pdf_report(similarity_score, resume_keywords, jd_keywords, overlap_data)
        st.download_button(
            label="Download Detailed Analysis Report (PDF)",
            data=pdf_buffer,
            file_name="resume_jd_analysis.pdf",
            mime="application/pdf",
            help="Download a PDF summary of the analysis results."
        )
    else:
        st.info("No analysis results available. Go to the 'Input' tab and click 'Analyze Compatibility' to start.")

# --- Display Visualizations Tab ---
with tab3:
    if st.session_state.analysis_results['has_run_analysis']:
        resume_processed = st.session_state.analysis_results['resume_processed']
        jd_processed = st.session_state.analysis_results['jd_processed']
        overlap_data = st.session_state.analysis_results['overlap_data']

        st.header("📈 Visualizations")

        # Word Clouds
        st.subheader("Keyword Insights")
        st.write("Visualizing the most frequent keywords in your resume and the job description.")
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            resume_wordcloud = generate_wordcloud_image(resume_processed, "Your Resume Keywords")
            st.image(resume_wordcloud, caption="Resume Key Themes", use_container_width =True)
        with col_wc2:
            jd_wordcloud = generate_wordcloud_image(jd_processed, "Job Description Keywords")
            st.image(jd_wordcloud, caption="Job Description Requirements", use_container_width =True)

        # Keyword Overlap Bar Chart
        st.subheader("Keyword Overlap Analysis")
        st.write("See how often common keywords appear in your resume versus the job description.")
        overlap_chart_fig = create_overlap_bar_chart(overlap_data)
        if overlap_chart_fig:
            st.pyplot(overlap_chart_fig)
        else:
            st.info("No common keywords to visualize.")

        # Radar Chart for Skill Categories (Simulated)
        st.subheader("Skill Category Comparison (Simulated)")
        st.write("This radar chart provides a *simulated* comparison of your strengths in different skill categories against the job description's focus. For a real-world application, this would require advanced NLP to categorize skills from text.")
        categories = ["Technical Skills", "Experience", "Education", "Soft Skills", "Certifications"]
        # Simulate scores for demonstration (replace with actual NLP categorization in a real app)
        resume_scores = [np.random.uniform(0.5, 1.0) for _ in categories]
        jd_scores = [np.random.uniform(0.4, 0.9) for _ in categories] # JD might have slightly lower max as it lists requirements

        radar_chart_fig = create_radar_chart(categories, resume_scores, jd_scores)
        st.pyplot(radar_chart_fig)

    else:
        st.info("No analysis results available for visualizations. Go to the 'Input' tab and click 'Analyze Compatibility' to start.")

# --- Display Suggestions Tab ---
with tab4:
    if st.session_state.analysis_results['has_run_analysis']:
        similarity_score = st.session_state.analysis_results['similarity_score']
        jd_keywords = st.session_state.analysis_results['jd_keywords']
        overlap_data = st.session_state.analysis_results['overlap_data']
        similarity_threshold = st.session_state.similarity_threshold

        st.header("💡 Suggestions for Improvement")
        score_percentage = similarity_score * 100

        if score_percentage < similarity_threshold:
            st.warning("Your resume could be better aligned with this job description.")
            st.write(
                f"""
                Your current match score is **{score_percentage:.1f}%**, which is below the set threshold of **{similarity_threshold}%**.
                To increase your match score, consider the following strategic adjustments:

                **1. Target Keywords from the Job Description:**
                * Review the **Job Description Keywords** in the 'Results' or 'Visualizations' tab.
                * Identify terms and phrases that appear frequently in the JD but are less prominent or missing from your resume.
                * **Example JD Keywords to focus on:**
                    {", ".join([kw for kw, freq in jd_keywords[:5]]) if jd_keywords else "N/A"} (These are the top 5 from the JD.)

                **2. Enhance Overlapping Skills:**
                * Look at the **Overlapping Keywords** in the 'Results' or 'Visualizations' tab.
                * For skills where the JD has a higher frequency than your resume, expand on your experience with those skills.
                * **Example:** If 'Python' appears 10 times in the JD and 3 times in your resume, add more projects or experiences where you extensively used Python.

                **3. Quantify Your Achievements:**
                * Instead of just stating duties, use numbers and data to describe the impact of your work.
                * **Example:** Instead of "Managed a team," write "Managed a team of 5 engineers, leading to a 20% increase in project delivery efficiency."

                **4. Tailor Your Experience Descriptions:**
                * For each role on your resume, ensure your bullet points directly address the responsibilities and requirements listed in the job description.
                * Prioritize experiences that are most relevant to this specific role.

                **5. Review Soft Skills:**
                * Many job descriptions emphasize soft skills (e.g., communication, teamwork, problem-solving). Ensure you have examples that demonstrate these.

                **6. Proofread Thoroughly:**
                * Even minor typos or grammatical errors can detract from your professional image.
                """
            )
        else:
            st.success("Your resume aligns very well with this job description!")
            st.write(
                f"""
                Your current match score is **{score_percentage:.1f}%**, indicating a strong alignment with the job description.
                You're in a great position! Here are some final tips to ensure your application stands out:

                * **Ensure Clarity and Conciseness:** While you have the right keywords, make sure your descriptions are easy to read and understand.
                * **Highlight Unique Strengths:** What unique value do you bring that might not be captured by keywords alone? Consider adding a strong summary or objective.
                * **Prepare for Behavioral Questions:** Given your strong match, be ready to provide specific examples of how you've applied the skills and experiences mentioned.
                * **Network:** Reach out to people in the company or role to gain insights and potentially get a referral.
                """
            )
    else:
        st.info("No analysis results available for suggestions. Go to the 'Input' tab and click 'Analyze Compatibility' to start.")

st.markdown("---")
st.caption("Powered by Google Gemini and Streamlit")
st.caption("Made in India with 💓 by Abhimanyu Singh")