import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import json
import requests # Re-import requests for Gemini API calls
from dotenv import load_dotenv # Re-import load_dotenv for API key
import pypdf # For PDF parsing
from docx import Document # For DOCX parsing
import io # For handling file-like objects
import datetime # Import datetime for getting the current year
import markdown # Import the markdown library for recommendations
import spacy # Import spaCy for NLP-based skill extraction

# Load environment variables from .env file
load_dotenv()

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.secret_key = os.urandom(24) # Secret key for session management

# In-memory user storage (for demonstration purposes)
# In a real application, this would be a database like Firestore
users = {
    "testuser": "password123",
    "john.doe": "securepass"
}

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'} # Only these file types are allowed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Gemini API Key - Now loaded from .env for the chatbot
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# It's good practice to print a masked version of the key for debug, not the full key
print(f"DEBUG: Loaded GEMINI_API_KEY: '{GEMINI_API_KEY[:5]}...' (Length: {len(GEMINI_API_KEY)})")

# Context processor to inject current_year into all templates
@app.context_processor
def inject_current_year():
    """Injects the current year into all Jinja2 templates."""
    return {'current_year': datetime.datetime.now().year}

# Register a custom Jinja2 filter for Markdown
@app.template_filter('markdown')
def markdown_filter(text):
    """Converts Markdown text to HTML."""
    return markdown.markdown(text)

# Helper function to check allowed file extensions
def allowed_file(filename):
    """Checks if a file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file):
    """
    Extracts text content from a file based on its extension.
    Supports .txt, .pdf, and .docx files.
    """
    filename = file.filename
    file_extension = filename.rsplit('.', 1)[1].lower()
    text_content = ""

    if file_extension == 'txt':
        # Read plain text file
        text_content = file.read().decode('utf-8', errors='ignore')
    elif file_extension == 'pdf':
        # Read PDF file using pypdf
        try:
            reader = pypdf.PdfReader(io.BytesIO(file.read()))
            for page in reader.pages:
                text_content += page.extract_text() or "" # Handle potential None from extract_text
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return None
    elif file_extension == 'docx':
        # Read DOCX file using python-docx
        try:
            document = Document(io.BytesIO(file.read()))
            for paragraph in document.paragraphs:
                text_content += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
            return None
    return text_content

# --- Gemini API Call Function (Reintroduced for Chatbot ONLY) ---
def call_gemini_api_for_chatbot(prompt):
    """
    Makes a call to the Gemini API with the given prompt for the chatbot.
    Returns the generated text or None if an error occurs.
    """
    if not GEMINI_API_KEY:
        print("Gemini API Key is not set. Chatbot will not function.")
        return "Sorry, the AI chatbot is not configured. Please ensure the API key is set."

    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]}) # Corrected from previous chat_history.append

    payload = {"contents": chat_history}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    print(f"Attempting Gemini API call for chatbot with prompt: {prompt[:100]}...")

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()
        print(f"Gemini API Raw Response: {json.dumps(result, indent=2)}")

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"Gemini API Generated Text: {generated_text[:100]}...")
            return generated_text
        else:
            print(f"Gemini API returned an unexpected structure or no candidates. Full response: {json.dumps(result, indent=2)}")
            if result.get('promptFeedback') and result['promptFeedback'].get('blockReason'):
                print(f"Gemini API blocked content due to: {result['promptFeedback']['blockReason']}")
                return "The AI response was blocked due to content policy. Please try a different query."
            return "Sorry, I couldn't generate a response. The AI returned an unexpected format."
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Gemini API: {e}")
        return f"Sorry, I'm having trouble connecting to the AI. Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e}")
        return f"An unexpected error occurred with the AI. Error: {e}"

# --- NLP-BASED SKILL EXTRACTION & RULE-BASED RECOMMENDATIONS (UNCHANGED) ---

# Predefined skills for common job roles (static data)
JOB_ROLE_SKILLS = {
    "Software Engineer": [
        "Python", "Java", "C++", "JavaScript", "SQL", "Data Structures",
        "Algorithms", "Object-Oriented Programming", "Web Development",
        "Cloud Computing", "Version Control (Git)", "Problem Solving",
        "Communication", "Teamwork", "Agile Methodologies", "REST API",
        "Docker", "Kubernetes", "Microservices", "Unit Testing", "CI/CD"
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Deep Learning",
        "Statistical Modeling", "Data Visualization", "Data Wrangling",
        "Big Data Technologies", "Cloud Computing", "Communication",
        "Problem Solving", "Critical Thinking", "Business Acumen",
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Spark"
    ],
    "Business Analyst": [
        "Requirements Gathering", "Data Analysis", "SQL", "Process Modeling",
        "Stakeholder Management", "Communication", "Problem Solving",
        "Project Management", "Microsoft Excel", "Presentation Skills",
        "Critical Thinking", "Business Process Improvement", "UML", "Agile",
        "User Stories", "Jira", "Confluence"
    ],
    "Project Manager": [
        "Project Planning", "Risk Management", "Budget Management",
        "Stakeholder Management", "Team Leadership", "Communication",
        "Negotiation", "Problem Solving", "Agile Methodologies",
        "Scrum", "Time Management", "Decision Making", "Gantt Charts",
        "Resource Allocation", "Conflict Resolution"
    ]
}

# Flatten all possible skills into a set for efficient lookup
ALL_POSSIBLE_SKILLS_LOWER = {
    skill.lower() for skills_list in JOB_ROLE_SKILLS.values() for skill in skills_list
}
# Add some common soft skills that might not be explicitly in job roles
ALL_POSSIBLE_SKILLS_LOWER.update({
    "leadership", "adaptability", "creativity", "time management", "analytical skills",
    "attention to detail", "customer service", "negotiation", "public speaking"
})


def get_target_skills_static(job_role):
    """
    Retrieves target skills for a job role from a predefined static list.
    """
    return JOB_ROLE_SKILLS.get(job_role, ["Communication", "Problem Solving", "Teamwork", "Adaptability"])

def extract_skills_nlp(resume_text):
    """
    Extracts skills from resume text using spaCy's NLP capabilities
    combined with a keyword matching approach.
    """
    extracted = set()
    doc = nlp(resume_text.lower()) # Process the resume text with spaCy

    # Approach: Keyword matching against a comprehensive list of known skills
    for skill in ALL_POSSIBLE_SKILLS_LOWER:
        if skill in doc.text: # Check if the skill string is present in the document text
            # Capitalize for display, handle multi-word skills
            extracted.add(skill.title() if len(skill.split()) == 1 else ' '.join(word.capitalize() for word in skill.split()))

    return list(extracted)

def generate_recommendation_non_ai(extracted_skills, target_job_skills, selected_job_role):
    """
    Generates a basic career recommendation based on extracted and target skills.
    This is a simple rule-based replacement for AI-powered recommendations.
    """
    missing_skills = [
        skill for skill in target_job_skills
        if skill.lower() not in [s.lower() for s in extracted_skills]
    ]

    recommendation = f"## Personalized Career Recommendations for {selected_job_role}\n\n"
    recommendation += "Based on your current skills and the target role, here are some recommendations:\n\n"

    # 1. Specific Job Roles (very basic, could be expanded with more logic)
    recommendation += "### 1. Specific Job Roles\n"
    if len(missing_skills) < 3 and len(extracted_skills) > 5:
        recommendation += f"* You seem well-aligned for a **{selected_job_role}** role. Focus on highlighting your strengths in your applications.\n"
        # Suggest related roles based on the target role
        if selected_job_role == "Software Engineer":
            recommendation += "* Consider roles like **Full-Stack Developer** or **Backend Engineer**.\n"
        elif selected_job_role == "Data Scientist":
            recommendation += "* Explore positions such as **Machine Learning Engineer** or **Data Analyst**.\n"
        elif selected_job_role == "Business Analyst":
            recommendation += "* Look into roles like **Systems Analyst** or **Product Owner**.\n"
        elif selected_job_role == "Project Manager":
            recommendation += "* Roles like **Program Manager** or **Scrum Master** might also be a good fit.\n"
    else:
        recommendation += "* Given your current skill set, you might also find opportunities in roles that require a broader range of skills, or roles that align with your strongest extracted skills.\n"
        recommendation += f"* Consider exploring roles related to: {', '.join(extracted_skills[:3]) if extracted_skills else 'general tech/business roles'}.\n"
    recommendation += "\n"

    # 2. Skill Improvement Suggestions
    recommendation += "### 2. Skill Improvement Suggestions\n"
    if missing_skills:
        recommendation += "To bridge the gaps for your target role, focus on these skills:\n"
        for skill in missing_skills:
            suggestion = f"* **{skill}:** "
            if "Python" in skill or "SQL" in skill or "Machine Learning" in skill or "Java" in skill or "JavaScript" in skill:
                suggestion += "Consider online courses on Coursera, Udemy, or edX. Practice with coding challenges on LeetCode or HackerRank."
            elif "Communication" in skill or "Problem Solving" in skill or "Leadership" in skill or "Management" in skill:
                suggestion += "Join Toastmasters, participate in group projects, or take online courses on soft skills. Look for relevant certifications."
            else:
                suggestion += "Search for online tutorials, workshops, or practical projects related to this skill."
            recommendation += suggestion + "\n"
    else:
        recommendation += "* Great job! You possess many of the target skills. Continue to refine your expertise in these areas.\n"
    recommendation += "\n"

    # 3. General Career Advice
    recommendation += "### 3. General Career Advice\n"
    recommendation += "* **Build a Strong Portfolio:** Showcase your projects and practical experience, especially for technical roles.\n"
    recommendation += "* **Networking:** Connect with professionals in your target industry on LinkedIn and attend industry events.\n"
    recommendation += "* **Continuous Learning:** The job market evolves rapidly. Always be open to learning new technologies and methodologies.\n"
    recommendation += "* **Tailor Your Resume:** Always customize your resume and cover letter for each specific job application.\n"
    recommendation += "\n"

    return recommendation

# --- END OF NLP-BASED SKILL EXTRACTION & RULE-BASED RECOMMENDATIONS ---

# --- Routes ---

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles new user registration."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists. Please choose a different one.', 'warning')
        else:
            users[username] = password
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logs out the current user."""
    session.pop('user', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    """Renders the user dashboard."""
    if 'user' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handles resume upload and initiates skill extraction and analysis."""
    if 'user' not in session:
        flash('Please log in to upload a resume.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        target_job_role = request.form.get('target_job_role')
        if not target_job_role:
            flash('Please select a target job role.', 'danger')
            return redirect(request.url)

        if 'resume' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['resume']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            resume_text = extract_text_from_file(file)

            if not resume_text:
                flash('Could not extract text from the uploaded file. Please ensure it is a valid TXT, PDF, or DOCX.', 'danger')
                return redirect(request.url)
            
            # Add a check for very short or empty resume text after extraction
            if not resume_text.strip():
                flash('Extracted resume text is empty. Please ensure your resume contains readable text.', 'danger')
                return redirect(request.url)

            # --- NLP-BASED LOGIC FOR SKILL EXTRACTION AND TARGET SKILLS ---
            target_job_skills = get_target_skills_static(target_job_role) # Using static list for target skills
            extracted_skills = extract_skills_nlp(resume_text) # Using spaCy for extraction

            if not target_job_skills:
                flash("Could not determine target job skills. Please try again.", 'danger')
                return redirect(request.url)
            
            if not extracted_skills:
                flash('Could not extract skills from resume using NLP/keyword matching. Please ensure your resume contains clear skill mentions.', 'danger')
                # Fallback to an empty list if no skills are found
                extracted_skills = []


            # 3. Calculate Resume Score and prepare Chart Data
            matched_skills_count = 0
            chart_labels = []
            chart_values = []
            
            # Use a set for faster lookups of extracted skills (case-insensitive)
            extracted_skills_lower = {s.lower() for s in extracted_skills}

            for skill in target_job_skills:
                chart_labels.append(skill)
                if skill.lower() in extracted_skills_lower:
                    chart_values.append(1) # Matched
                    matched_skills_count += 1
                else:
                    chart_values.append(0) # Missing

            resume_score = 0
            if target_job_skills:
                resume_score = (matched_skills_count / len(target_job_skills)) * 100
                resume_score = round(resume_score, 2) # Round to 2 decimal places

            # Store results in session
            session['chart_data'] = {
                'labels': chart_labels,
                'values': chart_values
            }
            session['extracted_skills'] = extracted_skills
            session['target_job_skills'] = target_job_skills
            session['resume_score'] = resume_score
            session['selected_job_role'] = target_job_role # Store the selected role

            flash('Resume analyzed and score calculated!', 'success')
            return redirect(url_for('charts')) # Redirect to charts to display analysis
        else:
            flash('Invalid file type. Allowed types: TXT, PDF, DOCX', 'danger')
    return render_template('upload.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Handles career chatbot interactions."""
    if 'user' not in session:
        flash('Please log in to use the chatbot.', 'warning')
        return redirect(url_for('login'))

    response_text = None
    user_query = ""
    if request.method == 'POST':
        user_query = request.form['query']
        if user_query:
            # --- PROFESSIONAL CHATBOT RESPONSE (GEMINI API) ---
            # This is the key change: calling the Gemini API for the chatbot's response
            full_prompt = f"You are a helpful and encouraging career guidance AI. Your goal is to provide supportive and actionable advice based on the user's query. Respond to the following query: {user_query}"
            response_text = call_gemini_api_for_chatbot(full_prompt)
            
            if not response_text: # Handle cases where API call failed or returned empty
                response_text = "Sorry, I couldn't generate a response at this time. There might be an issue with the AI service or content policy. Please try again later."
                flash("Error getting response from AI chatbot.", 'danger')
        else:
            flash('Please enter a query.', 'danger')
    return render_template('chatbot.html', response=response_text, user_query=user_query)

@app.route('/charts')
def charts():
    """Displays skill comparison charts and resume score."""
    if 'user' not in session:
        flash('Please log in to view charts.', 'warning')
        return redirect(url_for('login'))

    chart_data = session.get('chart_data', {'labels': [], 'values': []})
    resume_score = session.get('resume_score', 0)
    selected_job_role = session.get('selected_job_role', 'N/A')
    extracted_skills = session.get('extracted_skills', [])
    target_job_skills = session.get('target_job_skills', [])

    # Manually serialize data to JSON strings before passing to template
    chart_labels_json = json.dumps(chart_data['labels'])
    chart_values_json = json.dumps(chart_data['values'])
    
    # Generate colors based on values: green for 1 (matched), red for 0 (missing)
    background_colors = ["#10B981" if val == 1 else "#EF4444" for val in chart_data['values']] # Tailwind green-500, red-500
    background_colors_json = json.dumps(background_colors) # Serialize background colors too

    return render_template('charts.html',
                           chart_data=chart_data, # Keep original for other uses if any
                           chart_labels_json=chart_labels_json,
                           chart_values_json=chart_values_json,
                           background_colors_json=background_colors_json,
                           resume_score=resume_score,
                           selected_job_role=selected_job_role,
                           extracted_skills=extracted_skills,
                           target_job_skills=target_job_skills)

@app.route('/career_recommendation')
def career_recommendation():
    """Generates and displays career recommendations based on user's skills."""
    if 'user' not in session:
        flash('Please log in to get career recommendations.', 'warning')
        return redirect(url_for('login'))

    extracted_skills = session.get('extracted_skills', [])
    target_job_skills = session.get('target_job_skills', [])
    selected_job_role = session.get('selected_job_role', 'a job role')

    # --- NON-AI RECOMMENDATION GENERATION (RULE-BASED) ---
    recommendation_text = generate_recommendation_non_ai(extracted_skills, target_job_skills, selected_job_role)
    
    return render_template('career_recommendation.html', recommendation_text=recommendation_text)


@app.route('/resources')
def resources():
    """Renders the learning resources page."""
    if 'user' not in session:
        flash('Please log in to view resources.', 'warning')
        return redirect(url_for('login'))
    
    # You could potentially make this dynamic based on missing skills from session
    # For now, it will be a static list of general resources
    return render_template('resources.html')

if __name__ == '__main__':
    app.run(debug=True)
