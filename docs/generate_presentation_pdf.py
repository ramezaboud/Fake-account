"""
Generate PDF presentation for Fake Account Detection Project.
This script creates a comprehensive PDF document for project discussion.
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem
)
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pathlib import Path

# Colors
PRIMARY_COLOR = HexColor('#2563eb')  # Blue
SECONDARY_COLOR = HexColor('#059669')  # Green
ACCENT_COLOR = HexColor('#dc2626')  # Red
LIGHT_GREY = HexColor('#f3f4f6')
DARK_GREY = HexColor('#374151')

def create_styles():
    """Create custom styles for the PDF."""
    styles = getSampleStyleSheet()
    
    # Title style
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    
    # Section header
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=PRIMARY_COLOR,
        spaceBefore=20,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    ))
    
    # Subsection header
    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=SECONDARY_COLOR,
        spaceBefore=15,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))
    
    # Body text
    styles.add(ParagraphStyle(
        name='BodyText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=DARK_GREY,
        spaceBefore=6,
        spaceAfter=6,
        leading=16
    ))
    
    # Code style
    styles.add(ParagraphStyle(
        name='Code',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Courier',
        textColor=black,
        backColor=LIGHT_GREY,
        spaceBefore=4,
        spaceAfter=4,
        leftIndent=10,
        rightIndent=10
    ))
    
    # Bullet style
    styles.add(ParagraphStyle(
        name='BulletText',
        parent=styles['Normal'],
        fontSize=11,
        textColor=DARK_GREY,
        leftIndent=20,
        spaceBefore=3,
        spaceAfter=3
    ))
    
    return styles

def create_table_style():
    """Create table style."""
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GREY),
        ('TEXTCOLOR', (0, 1), (-1, -1), DARK_GREY),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ])

def build_pdf(output_path: str):
    """Build the PDF document."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = create_styles()
    story = []
    
    # ==================== TITLE PAGE ====================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("🤖 Fake Account Detection", styles['CustomTitle']))
    story.append(Paragraph("Machine Learning Project", styles['CustomTitle']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Detecting Fake/Bot Accounts on Social Media", styles['BodyText']))
    story.append(Paragraph("Using Random Forest Classification", styles['BodyText']))
    story.append(Spacer(1, 1*inch))
    
    # Project info table
    info_data = [
        ['Project Information', ''],
        ['Algorithm', 'Random Forest Classifier'],
        ['Accuracy', '92.69%'],
        ['ROC AUC', '98.47%'],
        ['Features Used', '29 Features'],
        ['Dataset Size', '21,060 Accounts'],
    ]
    info_table = Table(info_data, colWidths=[3*inch, 3*inch])
    info_table.setStyle(create_table_style())
    story.append(info_table)
    
    story.append(PageBreak())
    
    # ==================== PROBLEM STATEMENT ====================
    story.append(Paragraph("1. Problem Statement", styles['SectionHeader']))
    story.append(Paragraph(
        "Social media platforms like Twitter are plagued by millions of fake accounts (bots) that are used for:",
        styles['BodyText']
    ))
    
    problems = [
        "• Spreading fake news and misinformation",
        "• Manipulating public opinion during elections",
        "• Fraud and spam operations",
        "• Artificially inflating follower counts",
        "• Coordinated inauthentic behavior"
    ]
    for p in problems:
        story.append(Paragraph(p, styles['BulletText']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "<b>Solution:</b> A Machine Learning model that classifies accounts as <font color='green'>Real</font> or <font color='red'>Fake</font> "
        "based on profile characteristics and behavioral patterns.",
        styles['BodyText']
    ))
    
    # ==================== DATA SOURCES ====================
    story.append(Paragraph("2. Data Sources", styles['SectionHeader']))
    
    data_sources = [
        ['Source File', 'Description'],
        ['twibot20_clean.csv', 'TwiBot-20 Academic Dataset for bot detection'],
        ['synthetic_fake_accounts_*.csv', 'Synthetically generated fake accounts with known bot patterns'],
        ['users*.csv', 'Real user data collected from Twitter API'],
        ['labeled_dataset_new.csv', 'Final merged dataset (21,060 accounts)'],
    ]
    data_table = Table(data_sources, colWidths=[2.5*inch, 3.5*inch])
    data_table.setStyle(create_table_style())
    story.append(data_table)
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Dataset Statistics:</b>", styles['BodyText']))
    story.append(Paragraph("• Total Samples: 21,060 accounts", styles['BulletText']))
    story.append(Paragraph("• Training Set: 80% (16,848 samples)", styles['BulletText']))
    story.append(Paragraph("• Test Set: 20% (4,212 samples)", styles['BulletText']))
    story.append(Paragraph("• Label 0 = Real Account, Label 1 = Fake Account", styles['BulletText']))
    
    story.append(PageBreak())
    
    # ==================== FEATURE ENGINEERING ====================
    story.append(Paragraph("3. Feature Engineering (29 Features)", styles['SectionHeader']))
    
    # Basic Features
    story.append(Paragraph("A) Basic Count Features (5 features)", styles['SubsectionHeader']))
    basic_features = [
        ['Feature', 'Description'],
        ['statuses_count', 'Total number of tweets posted'],
        ['followers_count', 'Number of followers'],
        ['friends_count', 'Number of accounts being followed'],
        ['favourites_count', 'Number of likes given'],
        ['listed_count', 'Number of lists the account is added to'],
    ]
    basic_table = Table(basic_features, colWidths=[2*inch, 4*inch])
    basic_table.setStyle(create_table_style())
    story.append(basic_table)
    
    # Encoded Features
    story.append(Paragraph("B) Encoded Features (2 features)", styles['SubsectionHeader']))
    encoded_features = [
        ['Feature', 'Description'],
        ['sex_code', 'Gender inferred from name (-2=female to +2=male)'],
        ['lang_code', 'Encoded language code'],
    ]
    encoded_table = Table(encoded_features, colWidths=[2*inch, 4*inch])
    encoded_table.setStyle(create_table_style())
    story.append(encoded_table)
    
    # Activity Features
    story.append(Paragraph("C) Activity Features (2 features)", styles['SubsectionHeader']))
    activity_features = [
        ['Feature', 'Description'],
        ['tweets_per_day', 'Average tweets per day since account creation'],
        ['account_age_days', 'Age of account in days'],
    ]
    activity_table = Table(activity_features, colWidths=[2*inch, 4*inch])
    activity_table.setStyle(create_table_style())
    story.append(activity_table)
    
    # Profile Features
    story.append(Paragraph("D) Profile Features (3 features)", styles['SubsectionHeader']))
    profile_features = [
        ['Feature', 'Description'],
        ['description_length', 'Length of bio/description text'],
        ['default_profile', 'Whether using default profile settings (0/1)'],
        ['verified', 'Whether account is verified (0/1)'],
    ]
    profile_table = Table(profile_features, colWidths=[2*inch, 4*inch])
    profile_table.setStyle(create_table_style())
    story.append(profile_table)
    
    story.append(PageBreak())
    
    # Ratio Features (CRITICAL)
    story.append(Paragraph("E) Ratio Features - CRITICAL FOR DETECTION (6 features)", styles['SubsectionHeader']))
    story.append(Paragraph(
        "<font color='red'><b>These are the most important features for detecting fake accounts!</b></font>",
        styles['BodyText']
    ))
    
    ratio_features = [
        ['Feature', 'Description', 'Bot Pattern'],
        ['followers_to_friends_ratio', 'followers / friends', 'Very LOW (< 0.1)'],
        ['friends_to_followers_ratio', 'friends / followers', 'Very HIGH (> 10)'],
        ['engagement_ratio', 'favourites / statuses', 'Very LOW (near 0)'],
        ['reputation_score', 'followers / (followers + friends)', 'LOW (< 0.1)'],
        ['listed_ratio', 'listed_count / followers', 'Near 0'],
        ['favorites_per_tweet', 'favourites / statuses', 'Very LOW'],
    ]
    ratio_table = Table(ratio_features, colWidths=[2*inch, 2*inch, 2*inch])
    ratio_table.setStyle(create_table_style())
    story.append(ratio_table)
    
    # Name Features
    story.append(Paragraph("F) Name Features (6 features)", styles['SubsectionHeader']))
    name_features = [
        ['Feature', 'Description', 'Bot Pattern'],
        ['name_length', 'Length of display name', 'Very short or random'],
        ['screen_name_length', 'Length of username', 'Often long with numbers'],
        ['has_digits_in_name', 'Contains numbers in name', 'Often YES'],
        ['digit_ratio_in_screen_name', 'Ratio of digits in username', 'HIGH (user12345678)'],
        ['has_url', 'Has URL in profile', 'Usually NO'],
        ['has_description', 'Has bio text', 'Usually NO'],
    ]
    name_table = Table(name_features, colWidths=[2*inch, 2*inch, 2*inch])
    name_table.setStyle(create_table_style())
    story.append(name_table)
    
    # Suspicious Patterns
    story.append(Paragraph("G) Suspicious Pattern Features (5 features)", styles['SubsectionHeader']))
    suspicious_features = [
        ['Feature', 'Description'],
        ['is_new_account', 'Account less than 30 days old'],
        ['zero_followers', 'Has exactly 0 followers'],
        ['zero_statuses', 'Never posted any tweets'],
        ['following_many_no_followers', 'Following >100 but <10 followers'],
        ['high_friend_rate', 'Added many friends quickly (friends/age)'],
    ]
    suspicious_table = Table(suspicious_features, colWidths=[2.5*inch, 3.5*inch])
    suspicious_table.setStyle(create_table_style())
    story.append(suspicious_table)
    
    story.append(PageBreak())
    
    # ==================== DECISION LOGIC ====================
    story.append(Paragraph("4. How We Decide: Fake vs Real", styles['SectionHeader']))
    
    story.append(Paragraph("Key Indicators Comparison:", styles['SubsectionHeader']))
    
    comparison_data = [
        ['Indicator', 'Real Account', 'Fake Account (Bot)'],
        ['followers/friends ratio', '≈ 1.0 or higher', 'Very low (< 0.1)'],
        ['engagement_ratio', 'High (likes content)', 'Very low (no interaction)'],
        ['description', 'Detailed bio', 'Empty or generic'],
        ['tweets_per_day', 'Moderate (1-50)', 'Extreme (0 or >100)'],
        ['account_age', 'Usually older', 'Often new (< 30 days)'],
        ['digits in name', 'Rare', 'Common (user38291847)'],
        ['following pattern', 'Balanced', 'Follows many, few follow back'],
    ]
    comparison_table = Table(comparison_data, colWidths=[2*inch, 2*inch, 2*inch])
    comparison_table.setStyle(create_table_style())
    story.append(comparison_table)
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Example of a Suspicious Account:</b>", styles['BodyText']))
    
    example_code = """
    Suspicious Account Profile:
    ├── friends_count: 2,000 (following 2000 accounts)
    ├── followers_count: 5 (only 5 followers)
    ├── favourites_count: 0 (never liked anything)
    ├── statuses_count: 0 (never tweeted)
    ├── description: "" (empty bio)
    ├── screen_name: "bot_user_83627194"
    └── account_age: 10 days
    
    ═══════════════════════════════════════
    Result: 🔴 FAKE (Probability: 98%)
    ═══════════════════════════════════════
    """
    story.append(Paragraph(example_code.replace('\n', '<br/>'), styles['Code']))
    
    story.append(PageBreak())
    
    # ==================== MODEL PIPELINE ====================
    story.append(Paragraph("5. Model Pipeline Architecture", styles['SectionHeader']))
    
    pipeline_text = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                      ML PIPELINE ARCHITECTURE                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   Raw User Data (Twitter Profile)                               │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────────────────────────────┐                       │
    │   │     Step 1: FeatureEngineer         │                       │
    │   │     - Extract 29 features           │                       │
    │   │     - Calculate ratios              │                       │
    │   │     - Detect suspicious patterns    │                       │
    │   └─────────────────────────────────────┘                       │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────────────────────────────┐                       │
    │   │     Step 2: StandardScaler          │                       │
    │   │     - Normalize features            │                       │
    │   │     - Zero mean, unit variance      │                       │
    │   └─────────────────────────────────────┘                       │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────────────────────────────┐                       │
    │   │     Step 3: RandomForestClassifier  │                       │
    │   │     - n_estimators: 300 trees       │                       │
    │   │     - max_depth: 20                 │                       │
    │   │     - class_weight: balanced        │                       │
    │   └─────────────────────────────────────┘                       │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────────────────────────────┐                       │
    │   │     Output: Prediction              │                       │
    │   │     - 0: Real Account               │                       │
    │   │     - 1: Fake Account               │                       │
    │   │     - Probability scores            │                       │
    │   └─────────────────────────────────────┘                       │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
    """
    story.append(Paragraph(pipeline_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), styles['Code']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Why Random Forest?</b>", styles['SubsectionHeader']))
    story.append(Paragraph("• Handles mixed feature types (numerical + categorical)", styles['BulletText']))
    story.append(Paragraph("• Provides feature importance rankings", styles['BulletText']))
    story.append(Paragraph("• Robust to outliers and noisy data", styles['BulletText']))
    story.append(Paragraph("• Less prone to overfitting than single decision trees", styles['BulletText']))
    story.append(Paragraph("• No need for GPU - fast training and inference", styles['BulletText']))
    
    story.append(PageBreak())
    
    # ==================== MODEL RESULTS ====================
    story.append(Paragraph("6. Model Performance Results", styles['SectionHeader']))
    
    story.append(Paragraph("A) Overall Metrics:", styles['SubsectionHeader']))
    metrics_data = [
        ['Metric', 'Score', 'Interpretation'],
        ['Accuracy', '92.69%', 'Correctly classified 92.69% of all accounts'],
        ['Precision', '98.12%', 'When it says "Fake", it\'s right 98% of the time'],
        ['Recall', '86.66%', 'Catches 87% of all actual fake accounts'],
        ['F1-Score', '92.04%', 'Harmonic mean of precision and recall'],
        ['ROC AUC', '98.47%', 'Excellent discrimination ability'],
    ]
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.2*inch, 3.3*inch])
    metrics_table.setStyle(create_table_style())
    story.append(metrics_table)
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("B) Confusion Matrix (Test Set: 4,212 samples):", styles['SubsectionHeader']))
    
    confusion_data = [
        ['', 'Predicted Real', 'Predicted Fake'],
        ['Actual Real', 'TN = 2,124', 'FP = 34'],
        ['Actual Fake', 'FN = 274', 'TP = 1,780'],
    ]
    confusion_table = Table(confusion_data, colWidths=[2*inch, 2*inch, 2*inch])
    confusion_table.setStyle(create_table_style())
    story.append(confusion_table)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Interpretation:</b>", styles['BodyText']))
    story.append(Paragraph("• <b>True Positives (1,780):</b> Correctly identified fake accounts", styles['BulletText']))
    story.append(Paragraph("• <b>True Negatives (2,124):</b> Correctly identified real accounts", styles['BulletText']))
    story.append(Paragraph("• <b>False Positives (34):</b> Real accounts incorrectly marked as fake (Type I error)", styles['BulletText']))
    story.append(Paragraph("• <b>False Negatives (274):</b> Fake accounts that slipped through (Type II error)", styles['BulletText']))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("C) Decision Threshold:", styles['SubsectionHeader']))
    story.append(Paragraph(
        "The model uses a <b>decision threshold of 0.445</b>. If the probability of an account being fake "
        "is ≥ 44.5%, it's classified as <font color='red'>FAKE</font>. This threshold was tuned to balance "
        "precision and recall.",
        styles['BodyText']
    ))
    
    story.append(PageBreak())
    
    # ==================== API ENDPOINTS ====================
    story.append(Paragraph("7. REST API (FastAPI)", styles['SectionHeader']))
    
    story.append(Paragraph("The model is deployed as a REST API for easy integration:", styles['BodyText']))
    
    endpoints_data = [
        ['Method', 'Endpoint', 'Description'],
        ['GET', '/', 'API information'],
        ['GET', '/health', 'Health check status'],
        ['POST', '/predict', 'Predict single account'],
        ['POST', '/predict/batch', 'Predict multiple accounts'],
        ['GET', '/model/info', 'Get model information'],
    ]
    endpoints_table = Table(endpoints_data, colWidths=[1*inch, 2*inch, 3*inch])
    endpoints_table.setStyle(create_table_style())
    story.append(endpoints_table)
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Example API Request:</b>", styles['SubsectionHeader']))
    
    request_example = """
    POST /predict
    Content-Type: application/json
    
    {
      "user": {
        "screen_name": "test_user_123",
        "statuses_count": 100,
        "followers_count": 50,
        "friends_count": 2000,
        "favourites_count": 10,
        "listed_count": 0,
        "description": "",
        "created_at": "2025-12-01"
      }
    }
    """
    story.append(Paragraph(request_example.replace('\n', '<br/>'), styles['Code']))
    
    story.append(Paragraph("<b>Example API Response:</b>", styles['SubsectionHeader']))
    
    response_example = """
    {
      "prediction": 1,
      "label": "fake",
      "confidence": 0.89,
      "probabilities": {
        "Real": 0.11,
        "fake": 0.89
      }
    }
    """
    story.append(Paragraph(response_example.replace('\n', '<br/>'), styles['Code']))
    
    story.append(PageBreak())
    
    # ==================== PROJECT STRUCTURE ====================
    story.append(Paragraph("8. Project Structure", styles['SectionHeader']))
    
    structure_text = """
    fake-account/
    ├── app/
    │   └── api.py                  # FastAPI REST API
    ├── src/
    │   ├── feature_engineer.py     # Feature extraction (29 features)
    │   ├── train.py                # Model training with GridSearchCV
    │   ├── schemas.py              # Pydantic data models
    │   └── visualize.py            # Plotting utilities
    ├── models/
    │   ├── randomforest_pipeline.joblib   # Trained model
    │   └── eval_results.json              # Evaluation metrics
    ├── data/
    │   └── labeled_dataset_new.csv        # Training data
    ├── notebooks/
    │   └── FakeAccount.ipynb              # Exploratory analysis
    ├── docs/
    │   └── figures/                       # Evaluation plots
    ├── tests/
    │   ├── test_feature_engineer.py
    │   ├── test_model.py
    │   └── test_integration.py
    ├── requirements.txt
    └── README.md
    """
    story.append(Paragraph(structure_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), styles['Code']))
    
    story.append(PageBreak())
    
    # ==================== FAQ ====================
    story.append(Paragraph("9. Expected Questions & Answers", styles['SectionHeader']))
    
    qa_pairs = [
        ("Q1: Why Random Forest instead of Neural Network?",
         "A: The data is tabular/structured, Random Forest provides feature importance, doesn't require GPU, "
         "trains faster, and achieves excellent performance (98.5% AUC) on this type of data."),
        
        ("Q2: How do you handle imbalanced data?",
         "A: We use class_weight='balanced' in the classifier which automatically adjusts weights inversely "
         "proportional to class frequencies. We also use stratified train/test split to maintain class ratios."),
        
        ("Q3: Why isn't Recall 100%?",
         "A: Some sophisticated fake accounts mimic real user behavior very well. There's always a trade-off "
         "between Precision and Recall. We optimized for high Precision (98%) to minimize false accusations."),
        
        ("Q4: What are the most important features?",
         "A: The ratio-based features are most important: followers_to_friends_ratio, engagement_ratio, "
         "reputation_score, and the suspicious pattern flags like following_many_no_followers."),
        
        ("Q5: Can a real account be classified as fake?",
         "A: Yes, this is called a False Positive. Our model has only 34 FPs out of 4,212 test samples (0.8%), "
         "which means 98% Precision - very reliable."),
        
        ("Q6: How do you prevent overfitting?",
         "A: We use 5-fold cross-validation during hyperparameter tuning, limit max_depth to 20, "
         "use multiple trees (300 estimators), and evaluate on a held-out test set."),
        
        ("Q7: What if a new language appears?",
         "A: The FeatureEngineer maps unknown languages to a default 'unknown' code (0), ensuring the model "
         "can still make predictions for accounts using languages not seen during training."),
    ]
    
    for q, a in qa_pairs:
        story.append(Paragraph(f"<b>{q}</b>", styles['BodyText']))
        story.append(Paragraph(a, styles['BulletText']))
        story.append(Spacer(1, 0.15*inch))
    
    story.append(PageBreak())
    
    # ==================== HOW TO RUN ====================
    story.append(Paragraph("10. How to Run the Project", styles['SectionHeader']))
    
    story.append(Paragraph("<b>Installation:</b>", styles['SubsectionHeader']))
    install_code = """
    # Clone the repository
    git clone https://github.com/ramezaboud/Fake-account.git
    cd Fake-account
    
    # Create virtual environment
    python -m venv venv
    venv\\Scripts\\activate  # Windows
    
    # Install dependencies
    pip install -r requirements.txt
    """
    story.append(Paragraph(install_code.replace('\n', '<br/>'), styles['Code']))
    
    story.append(Paragraph("<b>Train the Model:</b>", styles['SubsectionHeader']))
    train_code = """
    python src/train.py --data_path data/labeled_dataset_new.csv --output_path models/
    """
    story.append(Paragraph(train_code.replace('\n', '<br/>'), styles['Code']))
    
    story.append(Paragraph("<b>Run the API:</b>", styles['SubsectionHeader']))
    api_code = """
    uvicorn app.api:app --reload --port 8000
    
    # Then open: http://localhost:8000/docs for Swagger UI
    """
    story.append(Paragraph(api_code.replace('\n', '<br/>'), styles['Code']))
    
    story.append(Paragraph("<b>Run Tests:</b>", styles['SubsectionHeader']))
    test_code = """
    pytest tests/ -v
    """
    story.append(Paragraph(test_code.replace('\n', '<br/>'), styles['Code']))
    
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Good Luck! 🎓", styles['CustomTitle']))
    
    # Build the PDF
    doc.build(story)
    print(f"PDF generated successfully: {output_path}")

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_path = output_dir / "Fake_Account_Detection_Presentation.pdf"
    build_pdf(str(output_path))
