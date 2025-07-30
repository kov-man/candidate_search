#!/usr/bin/env python3
"""
Create detailed resumes with GCP and modern cloud skills
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

# Create Resumes directory if it doesn't exist
if not os.path.exists('Resumes'):
    os.makedirs('Resumes')

# Detailed candidate profiles with GCP and modern skills
CANDIDATES = [
    {
        "name": "David Chen",
        "title": "Senior Cloud Architect & GCP Specialist",
        "email": "david.chen@cloudarchitect.com",
        "phone": "+1-555-0140",
        "location": "San Francisco, CA",
        "linkedin": "linkedin.com/in/david-chen-gcp",
        "github": "github.com/davidchen-cloud",
        "website": "davidchen.cloud",
        "summary": "Highly experienced Cloud Architect with 8+ years specializing in Google Cloud Platform (GCP) architecture, migration, and optimization. Led successful cloud transformations for Fortune 500 companies, reducing infrastructure costs by 40% while improving performance and scalability. Expert in designing multi-cloud solutions, implementing DevOps practices, and building scalable microservices architectures.",
        "experience": [
            {
                "title": "Senior Cloud Architect",
                "company": "CloudScale Solutions",
                "duration": "2021 - Present",
                "description": "Lead GCP architecture design and implementation for enterprise clients. Designed and deployed scalable cloud-native applications using GKE, Cloud Run, and Cloud Functions. Implemented CI/CD pipelines with Cloud Build and managed infrastructure with Terraform. Reduced client infrastructure costs by 35% through optimization and automation."
            },
            {
                "title": "DevOps Engineer",
                "company": "TechCorp Inc.",
                "duration": "2019 - 2021",
                "description": "Managed hybrid cloud infrastructure with GCP and on-premises systems. Implemented monitoring and logging solutions using Stackdriver and Cloud Monitoring. Automated deployment processes and reduced deployment time by 60%. Led migration of legacy applications to GCP."
            },
            {
                "title": "Software Engineer",
                "company": "StartupXYZ",
                "duration": "2017 - 2019",
                "description": "Developed cloud-native applications using Python, Node.js, and Go. Implemented microservices architecture with Docker and Kubernetes. Built and maintained CI/CD pipelines with Jenkins and GitHub Actions."
            }
        ],
        "skills": [
            "GCP", "Google Cloud Platform", "Kubernetes", "Docker", "Terraform", "Python", "Go", "Node.js",
            "Cloud Run", "Cloud Functions", "BigQuery", "Cloud Storage", "Cloud SQL", "Pub/Sub",
            "Cloud Build", "Cloud Monitoring", "Stackdriver", "Istio", "Anthos", "DevOps", "CI/CD",
            "Microservices", "API Gateway", "Load Balancing", "Auto Scaling", "Security", "IAM",
            "VPC", "Cloud Armor", "Cloud CDN", "Dataflow", "Dataproc", "Machine Learning", "AI"
        ],
        "education": "Master's in Computer Science, Stanford University",
        "certifications": [
            "Google Cloud Professional Cloud Architect",
            "Google Cloud Professional Data Engineer",
            "Google Cloud Professional DevOps Engineer",
            "Kubernetes Administrator (CKA)",
            "Terraform Associate"
        ]
    },
    {
        "name": "Sarah Rodriguez",
        "title": "Data Engineer & GCP Analytics Specialist",
        "email": "sarah.rodriguez@dataanalytics.com",
        "phone": "+1-555-0141",
        "location": "Austin, TX",
        "linkedin": "linkedin.com/in/sarah-rodriguez-data",
        "github": "github.com/sarahrodriguez-data",
        "website": "sarahrodriguez.analytics",
        "summary": "Experienced Data Engineer with 6+ years specializing in GCP data analytics and big data processing. Expert in designing and implementing data pipelines, data warehouses, and real-time analytics solutions. Successfully migrated enterprise data infrastructure to GCP, improving data processing speed by 300% and reducing costs by 50%.",
        "experience": [
            {
                "title": "Senior Data Engineer",
                "company": "DataFlow Analytics",
                "duration": "2020 - Present",
                "description": "Lead data engineering initiatives using GCP services including BigQuery, Dataflow, and Pub/Sub. Designed and implemented real-time data pipelines processing 10TB+ daily. Built data warehouse solutions and created dashboards with Data Studio. Implemented data governance and security policies."
            },
            {
                "title": "Data Analyst",
                "company": "AnalyticsCorp",
                "duration": "2018 - 2020",
                "description": "Developed ETL pipelines using Apache Beam and Dataflow. Created data models and dashboards for business intelligence. Implemented data quality monitoring and alerting systems. Collaborated with data scientists on ML model deployment."
            },
            {
                "title": "Software Engineer",
                "company": "TechStartup",
                "duration": "2016 - 2018",
                "description": "Built data processing applications using Python and Java. Implemented REST APIs and database solutions. Worked with cloud infrastructure and containerization technologies."
            }
        ],
        "skills": [
            "GCP", "BigQuery", "Dataflow", "Pub/Sub", "Cloud Storage", "Dataproc", "Data Studio",
            "Python", "Java", "SQL", "Apache Beam", "Apache Airflow", "Kafka", "Spark",
            "Machine Learning", "TensorFlow", "AutoML", "Data Engineering", "ETL", "Data Modeling",
            "Data Warehouse", "Real-time Analytics", "Data Governance", "Data Security", "IAM",
            "Cloud Composer", "Cloud Functions", "Cloud Run", "Data Catalog", "Dataprep"
        ],
        "education": "Bachelor's in Data Science, University of Texas",
        "certifications": [
            "Google Cloud Professional Data Engineer",
            "Google Cloud Professional Cloud Architect",
            "Apache Spark Developer",
            "Data Engineering on Google Cloud Platform"
        ]
    },
    {
        "name": "Michael Thompson",
        "title": "Site Reliability Engineer & GCP Infrastructure Expert",
        "email": "michael.thompson@sretech.com",
        "phone": "+1-555-0142",
        "location": "Seattle, WA",
        "linkedin": "linkedin.com/in/michael-thompson-sre",
        "github": "github.com/michaelthompson-sre",
        "website": "michaelthompson.sre",
        "summary": "Senior SRE with 7+ years experience in building and maintaining highly available, scalable systems on GCP. Expert in infrastructure automation, monitoring, and incident response. Led teams responsible for 99.99% uptime across multiple cloud regions. Implemented comprehensive observability solutions and automated incident response systems.",
        "experience": [
            {
                "title": "Senior Site Reliability Engineer",
                "company": "ReliabilityFirst",
                "duration": "2021 - Present",
                "description": "Lead SRE team managing infrastructure across multiple GCP regions. Implemented comprehensive monitoring with Cloud Monitoring, Logging, and Error Reporting. Designed and deployed highly available systems using GKE and Cloud Run. Automated incident response and reduced MTTR by 70%."
            },
            {
                "title": "DevOps Engineer",
                "company": "CloudOps Inc.",
                "duration": "2019 - 2021",
                "description": "Managed Kubernetes clusters and containerized applications. Implemented CI/CD pipelines with Cloud Build and ArgoCD. Built monitoring and alerting systems using Prometheus and Grafana. Automated infrastructure provisioning with Terraform."
            },
            {
                "title": "Systems Administrator",
                "company": "TechEnterprise",
                "duration": "2017 - 2019",
                "description": "Managed Linux servers and network infrastructure. Implemented backup and disaster recovery solutions. Automated system administration tasks using Python and Bash scripts."
            }
        ],
        "skills": [
            "GCP", "Kubernetes", "Docker", "Terraform", "Cloud Monitoring", "Cloud Logging",
            "Python", "Go", "Bash", "Linux", "Prometheus", "Grafana", "Istio", "Anthos",
            "Site Reliability Engineering", "SRE", "Infrastructure as Code", "CI/CD",
            "Cloud Build", "Cloud Run", "GKE", "Load Balancing", "Auto Scaling",
            "Disaster Recovery", "Backup", "Security", "IAM", "VPC", "Cloud Armor",
            "Error Reporting", "Trace", "Profiler", "Debugger"
        ],
        "education": "Bachelor's in Computer Engineering, University of Washington",
        "certifications": [
            "Google Cloud Professional Cloud Architect",
            "Google Cloud Professional DevOps Engineer",
            "Kubernetes Administrator (CKA)",
            "Site Reliability Engineering"
        ]
    },
    {
        "name": "Emily Zhang",
        "title": "Machine Learning Engineer & GCP AI Specialist",
        "email": "emily.zhang@mlengineering.com",
        "phone": "+1-555-0143",
        "location": "Boston, MA",
        "linkedin": "linkedin.com/in/emily-zhang-ml",
        "github": "github.com/emilyzhang-ml",
        "website": "emilyzhang.ml",
        "summary": "ML Engineer with 5+ years experience in developing and deploying machine learning models on GCP. Expert in AutoML, Vertex AI, and custom model development. Successfully deployed ML models serving 1M+ predictions daily. Specialized in computer vision, NLP, and recommendation systems.",
        "experience": [
            {
                "title": "Senior ML Engineer",
                "company": "AI Innovations",
                "duration": "2020 - Present",
                "description": "Lead ML model development and deployment using GCP Vertex AI and AutoML. Built recommendation systems processing millions of user interactions. Implemented MLOps pipelines with Kubeflow and Cloud Build. Achieved 95% model accuracy and 99.9% uptime for production models."
            },
            {
                "title": "Data Scientist",
                "company": "MLCorp",
                "duration": "2018 - 2020",
                "description": "Developed machine learning models for predictive analytics. Used TensorFlow and PyTorch for model development. Implemented data preprocessing pipelines and feature engineering. Collaborated with engineering teams on model deployment."
            },
            {
                "title": "Research Assistant",
                "company": "MIT AI Lab",
                "duration": "2016 - 2018",
                "description": "Conducted research in computer vision and natural language processing. Published papers on deep learning and neural networks. Developed experimental ML models and algorithms."
            }
        ],
        "skills": [
            "GCP", "Vertex AI", "AutoML", "TensorFlow", "PyTorch", "Python", "Kubeflow",
            "Machine Learning", "Deep Learning", "Computer Vision", "NLP", "Recommendation Systems",
            "MLOps", "Model Deployment", "Feature Engineering", "Data Preprocessing",
            "Cloud Functions", "Cloud Run", "BigQuery", "Cloud Storage", "Pub/Sub",
            "Model Serving", "A/B Testing", "Model Monitoring", "Explainable AI",
            "Neural Networks", "Transfer Learning", "Dataflow", "Dataproc"
        ],
        "education": "Master's in Computer Science, MIT",
        "certifications": [
            "Google Cloud Professional Machine Learning Engineer",
            "Google Cloud Professional Data Engineer",
            "TensorFlow Developer Certificate",
            "Deep Learning Specialization"
        ]
    },
    {
        "name": "Alex Johnson",
        "title": "Full Stack Developer & GCP Application Specialist",
        "email": "alex.johnson@fullstack.com",
        "phone": "+1-555-0144",
        "location": "New York, NY",
        "linkedin": "linkedin.com/in/alex-johnson-fullstack",
        "github": "github.com/alexjohnson-fullstack",
        "website": "alexjohnson.dev",
        "summary": "Full Stack Developer with 6+ years experience building scalable web applications on GCP. Expert in modern web technologies, cloud-native development, and serverless architectures. Led development teams and delivered projects on time and within budget. Specialized in React, Node.js, and cloud-native application development.",
        "experience": [
            {
                "title": "Senior Full Stack Developer",
                "company": "WebScale Solutions",
                "duration": "2021 - Present",
                "description": "Lead development of cloud-native web applications using React, Node.js, and GCP services. Implemented serverless architectures with Cloud Functions and Cloud Run. Built scalable APIs and microservices. Achieved 99.9% uptime and sub-100ms response times."
            },
            {
                "title": "Frontend Developer",
                "company": "DigitalAgency",
                "duration": "2019 - 2021",
                "description": "Developed responsive web applications using React, TypeScript, and modern CSS. Implemented state management with Redux and built reusable component libraries. Collaborated with UX/UI designers and backend developers."
            },
            {
                "title": "Backend Developer",
                "company": "TechStartup",
                "duration": "2017 - 2019",
                "description": "Built REST APIs and microservices using Node.js and Python. Implemented database solutions and authentication systems. Deployed applications using Docker and cloud platforms."
            }
        ],
        "skills": [
            "GCP", "Cloud Run", "Cloud Functions", "App Engine", "Firebase", "React",
            "Node.js", "TypeScript", "JavaScript", "Python", "Docker", "Kubernetes",
            "Full Stack Development", "Web Development", "API Development", "Microservices",
            "Serverless", "Cloud Native", "REST APIs", "GraphQL", "Authentication",
            "Cloud Storage", "Cloud SQL", "Firestore", "Pub/Sub", "Cloud Build",
            "CI/CD", "Git", "GitHub", "Agile", "Scrum"
        ],
        "education": "Bachelor's in Computer Science, NYU",
        "certifications": [
            "Google Cloud Professional Cloud Developer",
            "Google Cloud Professional Cloud Architect",
            "React Developer Certification",
            "Node.js Developer Certification"
        ]
    }
]

def create_resume(candidate):
    """Create a detailed resume PDF for a candidate."""
    filename = f"Resumes/{candidate['name'].replace(' ', '_').lower()}_resume.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    # Header
    story.append(Paragraph(candidate['name'], title_style))
    story.append(Paragraph(candidate['title'], styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Contact Information
    contact_info = [
        f"Email: {candidate['email']}",
        f"Phone: {candidate['phone']}",
        f"Location: {candidate['location']}",
        f"LinkedIn: {candidate['linkedin']}",
        f"GitHub: {candidate['github']}",
        f"Website: {candidate['website']}"
    ]
    
    for contact in contact_info:
        story.append(Paragraph(contact, styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", heading_style))
    story.append(Paragraph(candidate['summary'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Experience
    story.append(Paragraph("PROFESSIONAL EXPERIENCE", heading_style))
    
    for exp in candidate['experience']:
        story.append(Paragraph(f"<b>{exp['title']}</b> - {exp['company']}", styles['Heading3']))
        story.append(Paragraph(f"<i>{exp['duration']}</i>", styles['Normal']))
        story.append(Paragraph(exp['description'], styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Skills
    story.append(Paragraph("TECHNICAL SKILLS", heading_style))
    skills_text = ", ".join(candidate['skills'])
    story.append(Paragraph(skills_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Education
    story.append(Paragraph("EDUCATION", heading_style))
    story.append(Paragraph(candidate['education'], styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Certifications
    story.append(Paragraph("CERTIFICATIONS", heading_style))
    for cert in candidate['certifications']:
        story.append(Paragraph(f"• {cert}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"Created resume for {candidate['name']}: {filename}")

def main():
    """Create resumes for all candidates."""
    print("Creating detailed resumes with GCP and modern skills...")
    
    for candidate in CANDIDATES:
        create_resume(candidate)
    
    print(f"Successfully created {len(CANDIDATES)} detailed resumes!")
    print("Resumes saved in the 'Resumes' directory")

if __name__ == "__main__":
    main() 