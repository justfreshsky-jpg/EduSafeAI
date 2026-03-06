import collections
import hashlib
import logging
import os
from logging.handlers import RotatingFileHandler
os.environ.setdefault('HTTPX_PROXIES', 'null')  # Fix Render/httpx proxies bug
import requests
import threading
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

# ── LOGGING ──────────────────────────────────────────────────
_log_fmt = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])
_log_dir = os.environ.get('LOG_DIR')
if _log_dir:
    _file_handler = RotatingFileHandler(
        os.path.join(_log_dir, 'edusafeai.log'), maxBytes=5_000_000, backupCount=3
    )
    _file_handler.setFormatter(_log_fmt)
    logging.getLogger().addHandler(_file_handler)

app = Flask(__name__)

GROQ_KEY = os.environ.get('GROQ_KEY')
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# ── BLOG SCRAPER (background, server-side only) ──────────────
_cache = {"content": "", "last": 0}
_blog_lock = threading.Lock()

FALLBACK = """
[LESSON] Effective lessons require hooks, differentiation, formative checks, and exit tickets.
[IEP/ELL] ELL Proficiency levels: 1=Entering, 2=Emerging, 3=Developing, 4=Expanding, 5=Bridging (based on WIDA framework).
[STANDARDS] Common frameworks: Common Core (US), National Curriculum (UK), IB, Cambridge, state/provincial standards.
[POLICY] AI use policies must address plagiarism, data privacy, student safety, and academic integrity.
[504/IEP] 504 covers access accommodations; IEP covers specialized instruction under IDEA (US) or equivalent local law.
[EMAIL] Parent communications should be culturally responsive and available in families' home languages.
[FEEDBACK] Rubric-aligned feedback: Strength, Evidence, Reasoning, Next Step.
[UNIT] 2-week unit: overview, daily breakdown, formative + summative assessments.
"""


import json

# ── SHARED STANDARD SETS ────────────────────────────────────
_CCSS_ELA_STDS = {
    "K-2": [
        {"code":"RL.K.1","desc":"Reading Literature: With prompting, ask and answer questions about key details"},
        {"code":"RL.1.2","desc":"Reading Literature: Retell stories including key details and central message"},
        {"code":"RI.2.1","desc":"Reading Informational: Ask and answer questions about key details in a text"},
        {"code":"W.1.3","desc":"Writing: Write narratives recounting two or more events in sequence"},
        {"code":"SL.K.1","desc":"Speaking & Listening: Participate in collaborative conversations with peers"},
        {"code":"L.2.2","desc":"Language: Demonstrate command of capitalization, punctuation, and spelling"},
    ],
    "3-5": [
        {"code":"RL.3.1","desc":"Reading Literature: Ask and answer questions referring explicitly to the text"},
        {"code":"RL.4.2","desc":"Reading Literature: Determine a theme from details in the text"},
        {"code":"RI.5.8","desc":"Reading Informational: Explain how an author uses reasons and evidence"},
        {"code":"W.4.1","desc":"Writing: Write opinion pieces supporting a point of view with reasons"},
        {"code":"SL.3.4","desc":"Speaking & Listening: Report on a topic with appropriate facts and details"},
        {"code":"L.5.4","desc":"Language: Determine meaning of unknown words using context clues"},
    ],
    "6-8": [
        {"code":"RL.6.1","desc":"Reading Literature: Cite textual evidence to support analysis of the text"},
        {"code":"RL.7.2","desc":"Reading Literature: Determine a theme and analyze its development"},
        {"code":"RI.8.6","desc":"Reading Informational: Determine author's point of view or purpose"},
        {"code":"W.7.1","desc":"Writing: Write arguments to support claims with clear reasons and evidence"},
        {"code":"SL.6.2","desc":"Speaking & Listening: Interpret information presented in diverse media"},
        {"code":"L.8.1","desc":"Language: Demonstrate command of grammar — verbs, voice, mood"},
    ],
    "9-12": [
        {"code":"RL.9-10.1","desc":"Reading Literature: Cite strong and thorough textual evidence"},
        {"code":"RL.11-12.2","desc":"Reading Literature: Determine two or more themes and analyze development"},
        {"code":"RI.11-12.6","desc":"Reading Informational: Determine author's purpose with effective rhetoric"},
        {"code":"W.9-10.1","desc":"Writing: Write arguments to support claims in analysis of substantive topics"},
        {"code":"SL.11-12.4","desc":"Speaking & Listening: Present information clearly with logical reasoning"},
        {"code":"L.9-10.3","desc":"Language: Apply knowledge of language for comprehension and style"},
    ],
}
_CCSS_MATH_STDS = {
    "K-2": [
        {"code":"K.CC.A.1","desc":"Counting: Count to 100 by ones and tens"},
        {"code":"1.OA.A.1","desc":"Operations: Use addition and subtraction within 20"},
        {"code":"2.NBT.B.5","desc":"Number: Fluently add and subtract within 100"},
        {"code":"K.G.A.1","desc":"Geometry: Describe objects using names of shapes"},
        {"code":"1.MD.A.1","desc":"Measurement: Order three objects by length"},
        {"code":"2.OA.C.3","desc":"Operations: Determine odd or even number of objects"},
    ],
    "3-5": [
        {"code":"3.OA.A.1","desc":"Operations: Interpret products of whole numbers"},
        {"code":"4.NF.A.1","desc":"Fractions: Explain equivalent fractions using models"},
        {"code":"5.NBT.A.1","desc":"Number: Place value — digit is 10x the digit to its right"},
        {"code":"3.MD.C.7","desc":"Measurement: Relate area to multiplication and addition"},
        {"code":"4.G.A.1","desc":"Geometry: Draw and identify lines, angles, and shapes"},
        {"code":"5.NF.A.1","desc":"Fractions: Add and subtract fractions with unlike denominators"},
    ],
    "6-8": [
        {"code":"6.RP.A.1","desc":"Ratios: Understand concept of ratio and use ratio language"},
        {"code":"7.NS.A.1","desc":"Number System: Add and subtract rational numbers"},
        {"code":"8.EE.B.5","desc":"Expressions: Graph proportional relationships and interpret slope"},
        {"code":"6.G.A.1","desc":"Geometry: Find area of polygons by composing and decomposing"},
        {"code":"7.SP.A.1","desc":"Statistics: Understand that statistics can answer questions about data"},
        {"code":"8.F.A.1","desc":"Functions: Understand that a function assigns one output to each input"},
    ],
    "9-12": [
        {"code":"HSA.CED.A.1","desc":"Algebra: Create equations and inequalities to solve problems"},
        {"code":"HSF.IF.A.1","desc":"Functions: Understand that a function assigns exactly one output"},
        {"code":"HSS.ID.A.1","desc":"Statistics: Represent data with dot plots, histograms, box plots"},
        {"code":"HSG.CO.A.1","desc":"Geometry: Know precise definitions of angle, circle, line, parallel"},
        {"code":"HSN.RN.A.1","desc":"Number: Explain how rational exponents extend integer exponent properties"},
    ],
}
_NGSS_SCI_STDS = {
    "K-2": [
        {"code":"K-PS2-1","desc":"Physical Science: Plan investigation to compare effects of different forces"},
        {"code":"K-ESS2-1","desc":"Earth & Space Science: Use a model to represent the relationship between needs of plants and animals"},
        {"code":"1-LS1-1","desc":"Life Science: Use materials to design a solution to a human problem"},
        {"code":"2-PS1-1","desc":"Physical Science: Plan an investigation to describe and classify matter by properties"},
    ],
    "3-5": [
        {"code":"3-LS1-1","desc":"Life Science: Develop models to describe organisms' unique life cycles"},
        {"code":"4-PS4-1","desc":"Physical Science: Develop a model of waves to describe patterns of amplitude and wavelength"},
        {"code":"5-ESS1-1","desc":"Earth & Space: Support argument that the apparent brightness of the sun and stars is due to relative distances"},
        {"code":"4-LS1-1","desc":"Life Science: Construct argument that plants and animals have internal and external structures"},
    ],
    "6-8": [
        {"code":"MS-PS1-1","desc":"Physical Science: Develop models to describe atomic composition of simple molecules"},
        {"code":"MS-LS1-1","desc":"Life Science: Provide evidence that living things are made of cells"},
        {"code":"MS-ESS1-1","desc":"Earth & Space: Develop and use a model of the Earth-Sun-Moon system"},
        {"code":"MS-PS2-2","desc":"Physical Science: Plan an investigation to show that the change in motion depends on net force"},
        {"code":"MS-LS2-1","desc":"Life Science: Analyze and interpret data for patterns of interactions among organisms"},
    ],
    "9-12": [
        {"code":"HS-PS1-1","desc":"Chemistry: Use the periodic table to predict properties of elements"},
        {"code":"HS-LS1-1","desc":"Biology: Explain how DNA determines the structure of proteins"},
        {"code":"HS-ESS1-1","desc":"Earth & Space: Develop a model based on evidence to illustrate the life span of the sun"},
        {"code":"HS-PS2-1","desc":"Physics: Analyze data to support the claim that Newton's second law describes motion"},
        {"code":"HS-LS4-1","desc":"Biology: Communicate evidence that common ancestry is supported by multiple lines of evidence"},
    ],
}
_GENERIC_SS_STDS = {
    "K-2": [
        {"code":"SS.K.1","desc":"Community: Describe family and community roles and responsibilities"},
        {"code":"SS.1.2","desc":"Geography: Identify maps and globes as representations of Earth's surface"},
        {"code":"SS.2.1","desc":"Geography: Identify continents, oceans, and major landforms"},
        {"code":"SS.K.3","desc":"Civics: Explain the purpose of rules and laws in the community"},
    ],
    "3-5": [
        {"code":"SS.3.1","desc":"History: Describe significant events in local and state history"},
        {"code":"SS.4.2","desc":"Geography: Explain human-environment interaction and migration patterns"},
        {"code":"SS.5.3","desc":"Civics: Describe the structure of the US government and its branches"},
        {"code":"SS.4.4","desc":"Economics: Explain the basic concepts of supply, demand, and trade"},
    ],
    "6-8": [
        {"code":"SS.6.1","desc":"World History: Analyze early civilizations and their lasting contributions"},
        {"code":"SS.7.2","desc":"Geography: Analyze how human activities shape the physical environment"},
        {"code":"SS.8.3","desc":"US History: Analyze causes and effects of the American Revolution"},
        {"code":"SS.8.5","desc":"Civics: Evaluate the principles of democracy embedded in the Constitution"},
    ],
    "9-12": [
        {"code":"SS.10.2","desc":"Government: Analyze the principles of constitutional democracy"},
        {"code":"SS.11.3","desc":"US History: Evaluate Reconstruction and its effects on civil rights"},
        {"code":"SS.12.1","desc":"Economics: Analyze how market economies allocate resources"},
        {"code":"SS.10.5","desc":"World History: Evaluate the causes and consequences of global conflicts"},
    ],
}
_CCSS_FALLBACK = (
    "[LESSON] CCSS-aligned lessons require standards-based objectives, formative assessment, and differentiation.\n"
    "[ELA] CCSS ELA standards emphasize text complexity, evidence-based writing, and academic vocabulary.\n"
    "[MATH] CCSS Math standards focus on conceptual understanding, procedural fluency, and application.\n"
    "[SCIENCE] NGSS science standards emphasize three-dimensional learning: practices, crosscutting concepts, and disciplinary core ideas."
)

def _ccss_state(name, standards_name, assessment_name, edu_url, focus_prompt,
                assessment_system_prompt, assessment_result_label,
                assessment_example_placeholder="e.g. RL.8.1 or 8.EE.B.5 or leave blank"):
    return {
        "name": name,
        "standards_name": standards_name,
        "assessment_name": assessment_name,
        "standards_body": f"{name} Department of Education",
        "edu_url": edu_url,
        "focus_prompt": focus_prompt,
        "fallback": _CCSS_FALLBACK,
        "ela_standards": _CCSS_ELA_STDS,
        "math_standards": _CCSS_MATH_STDS,
        "science_label": "Science (NGSS)",
        "science_standards": _NGSS_SCI_STDS,
        "assessment_system_prompt": assessment_system_prompt,
        "assessment_standard_label": "Standard (e.g. RL.8.1 or 8.EE.B.5)",
        "assessment_example_placeholder": assessment_example_placeholder,
        "assessment_result_label": assessment_result_label,
    }

# ── STATE DATA ───────────────────────────────────────────────
STATE_DATA = {
    "worldwide": {
        "name": "Worldwide (CCSS/NGSS)",
        "standards_name": "Common Core (CCSS) / NGSS",
        "assessment_name": "NAEP",
        "standards_body": "International / CCSS / NGSS",
        "edu_url": "https://www.ed.gov/",
        "focus_prompt": "You are supporting K-12 educators anywhere in the world. Align all content to Common Core State Standards (CCSS) and NGSS science standards — internationally recognized frameworks used widely across the globe. Reference NAEP assessment data and WIDA frameworks where relevant. Provide practical, classroom-ready guidance with culturally responsive practices.",
        "fallback": "[LESSON] Common Core-aligned lessons require standards-based objectives, formative assessment, and differentiation.\n[ELA] CCSS ELA standards emphasize text complexity, evidence-based writing, and academic vocabulary.\n[MATH] CCSS Math standards focus on conceptual understanding, procedural fluency, and application.\n[SCIENCE] NGSS science standards emphasize three-dimensional learning: practices, crosscutting concepts, and disciplinary core ideas.\n[POLICY] ESSA requires states to set academic standards and measure student performance on state assessments.",
        "ela_standards": _CCSS_ELA_STDS,
        "math_standards": _CCSS_MATH_STDS,
        "science_label": "Science (NGSS)",
        "science_standards": _NGSS_SCI_STDS,
        "assessment_system_prompt": "Standards-based assessment specialist using CCSS and NAEP frameworks.",
        "assessment_standard_label": "Standard (e.g. RL.8.1 or 8.EE.B.5)",
        "assessment_example_placeholder": "e.g. RL.8.1 or leave blank",
        "assessment_result_label": "NAEP-Aligned Practice Questions",
    },
    "AL": _ccss_state("Alabama", "Alabama Courses of Study (ACOS)", "ACAP",
        "https://www.alabamaachieves.org/",
        "You are supporting Alabama K-12 educators. Align content to Alabama Courses of Study (ACOS) and ACAP (Alabama Comprehensive Assessment Program) assessments. Use CCSS-aligned codes for ELA and Math.",
        "ACAP assessment specialist aligned to Alabama Courses of Study (ACOS).",
        "ACAP Practice Questions"),
    "AK": _ccss_state("Alaska", "Alaska Academic Standards", "AK STAR",
        "https://education.alaska.gov/",
        "You are supporting Alaska K-12 educators. Align content to Alaska Academic Standards and AK STAR assessments. Use CCSS-aligned codes for ELA and Math.",
        "AK STAR assessment specialist aligned to Alaska Academic Standards.",
        "AK STAR Practice Questions"),
    "AZ": _ccss_state("Arizona", "Arizona Academic Standards", "AzMERIT",
        "https://www.azed.gov/",
        "You are supporting Arizona K-12 educators. Align content to Arizona Academic Standards and AzMERIT assessments. Use CCSS-aligned codes for ELA and Math.",
        "AzMERIT assessment specialist aligned to Arizona Academic Standards.",
        "AzMERIT Practice Questions"),
    "AR": _ccss_state("Arkansas", "Arkansas Academic Standards", "ATLAS",
        "https://dese.ade.arkansas.gov/",
        "You are supporting Arkansas K-12 educators. Align content to Arkansas Academic Standards and ATLAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "ATLAS assessment specialist aligned to Arkansas Academic Standards.",
        "ATLAS Practice Questions"),
    "CA": {
        "name": "California",
        "standards_name": "California CCSS / CA NGSS",
        "assessment_name": "CAASPP",
        "standards_body": "California Department of Education",
        "edu_url": "https://www.cde.ca.gov/",
        "focus_prompt": "You are supporting California K-12 educators. Align all content to California's Common Core State Standards and CA NGSS, assessed through CAASPP (California Assessment of Student Performance and Progress). Use CCSS codes for ELA and Math.",
        "fallback": _CCSS_FALLBACK,
        "ela_standards": _CCSS_ELA_STDS,
        "math_standards": _CCSS_MATH_STDS,
        "science_label": "Science (CA NGSS)",
        "science_standards": _NGSS_SCI_STDS,
        "assessment_system_prompt": "CAASPP assessment specialist aligned to California CCSS and CA NGSS.",
        "assessment_standard_label": "Standard (e.g. RL.8.1 or 8.EE.B.5)",
        "assessment_example_placeholder": "e.g. RL.8.1 or 8.EE.B.5 or leave blank",
        "assessment_result_label": "CAASPP Practice Questions",
    },
    "CO": _ccss_state("Colorado", "Colorado Academic Standards (CAS)", "CMAS",
        "https://www.cde.state.co.us/",
        "You are supporting Colorado K-12 educators. Align content to Colorado Academic Standards (CAS) and CMAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "CMAS assessment specialist aligned to Colorado Academic Standards (CAS).",
        "CMAS Practice Questions"),
    "CT": _ccss_state("Connecticut", "Connecticut Core Standards", "SBAC",
        "https://portal.ct.gov/SDE",
        "You are supporting Connecticut K-12 educators. Align content to Connecticut Core Standards and Smarter Balanced (SBAC) assessments. Use CCSS-aligned codes for ELA and Math.",
        "Smarter Balanced (SBAC) assessment specialist aligned to Connecticut Core Standards.",
        "CT SBAC Practice Questions"),
    "DE": _ccss_state("Delaware", "Delaware Content Standards", "DeSSA",
        "https://www.doe.k12.de.us/",
        "You are supporting Delaware K-12 educators. Align content to Delaware Content Standards and DeSSA assessments. Use CCSS-aligned codes for ELA and Math.",
        "DeSSA assessment specialist aligned to Delaware Content Standards.",
        "DeSSA Practice Questions"),
    "FL": {
        "name": "Florida",
        "standards_name": "Florida B.E.S.T. Standards",
        "assessment_name": "F.A.S.T.",
        "standards_body": "Florida Department of Education",
        "edu_url": "https://www.fldoe.org/",
        "focus_prompt": "You are supporting Florida K-12 educators. Align all content to Florida's B.E.S.T. (Benchmarks for Excellent Student Thinking) Standards and F.A.S.T. (Florida Assessment of Student Thinking) assessments. Use B.E.S.T. codes like ELA.3.R.1.1 for ELA and MA.5.AR.1.1 for Math.",
        "fallback": "[LESSON] Florida B.E.S.T. Standards use ELA.grade.R.strand.benchmark and MA.grade.domain.strand.benchmark codes.\n[ELA] Florida B.E.S.T. ELA standards use codes like ELA.3.R.1.1 emphasizing reading, vocabulary, and writing.\n[MATH] Florida B.E.S.T. Math standards use codes like MA.5.AR.1.1 emphasizing algebraic reasoning.\n[SCIENCE] Florida NGSSS science standards use codes like SC.7.N.1.1.",
        "ela_standards": {
            "K-2": [{"code":"ELA.K.R.1.1","desc":"Reading: Identify and explain key details in a text"},{"code":"ELA.1.R.1.1","desc":"Reading: Describe key details in a text"},{"code":"ELA.2.C.1.2","desc":"Writing: Write informational texts about topics"}],
            "3-5": [{"code":"ELA.3.R.1.1","desc":"Reading: Describe main idea and key details in texts"},{"code":"ELA.4.R.1.1","desc":"Reading: Analyze and explain key details and central idea"},{"code":"ELA.5.C.1.4","desc":"Writing: Write opinion pieces with supporting reasons"}],
            "6-8": [{"code":"ELA.6.R.1.1","desc":"Reading: Cite textual evidence to support analysis of text"},{"code":"ELA.7.R.1.1","desc":"Reading: Cite several pieces of textual evidence for analysis"},{"code":"ELA.8.R.1.1","desc":"Reading: Cite the textual evidence that most strongly supports analysis"}],
            "9-12": [{"code":"ELA.9.R.1.1","desc":"Reading: Cite strong and thorough textual evidence for analysis"},{"code":"ELA.10.C.1.4","desc":"Writing: Write arguments with precise claims and evidence"},{"code":"ELA.11.C.1.4","desc":"Writing: Write substantive arguments that establish significance"}],
        },
        "math_standards": {
            "K-2": [{"code":"MA.K.NSO.1.1","desc":"Number Sense: Count forward and backward to 20"},{"code":"MA.1.NSO.2.1","desc":"Number Sense: Recall addition and subtraction facts to 20"}],
            "3-5": [{"code":"MA.3.AR.1.1","desc":"Algebraic Reasoning: Apply properties of multiplication and division"},{"code":"MA.4.NSO.2.1","desc":"Number Sense: Multiply multi-digit whole numbers"},{"code":"MA.5.AR.1.1","desc":"Algebraic Reasoning: Solve multi-step problems using equations"}],
            "6-8": [{"code":"MA.6.AR.1.1","desc":"Algebraic Reasoning: Write and solve one-step equations"},{"code":"MA.7.AR.1.1","desc":"Algebraic Reasoning: Apply properties to simplify expressions"},{"code":"MA.8.AR.1.1","desc":"Algebraic Reasoning: Analyze and represent linear relationships"}],
            "9-12": [{"code":"MA.912.AR.1.1","desc":"Algebra: Solve linear equations and inequalities in one variable"},{"code":"MA.912.F.1.1","desc":"Functions: Identify and interpret key features of functions"},{"code":"MA.912.S.1.1","desc":"Statistics: Analyze and interpret data using statistical measures"}],
        },
        "science_label": "Science (NGSSS)",
        "science_standards": {
            "K-2": [{"code":"SC.K.N.1.1","desc":"Science Process: Collaborate with a partner to collect information"},{"code":"SC.1.E.5.1","desc":"Earth Science: Observe and describe patterns of stars in the night sky"}],
            "3-5": [{"code":"SC.3.N.1.1","desc":"Science Process: Raise questions and explore observable phenomena"},{"code":"SC.4.E.6.1","desc":"Earth Science: Identify the three categories of rocks"},{"code":"SC.5.P.8.1","desc":"Physical Science: Compare and contrast properties of matter"}],
            "6-8": [{"code":"SC.6.N.1.1","desc":"Science Process: Define a problem and use appropriate reference materials"},{"code":"SC.7.L.15.1","desc":"Life Science: Recognize that fossil evidence is consistent with evolution"},{"code":"SC.8.P.8.1","desc":"Physical Science: Examine and compare properties of matter"}],
            "9-12": [{"code":"SC.912.N.1.1","desc":"Science Process: Define a problem based on specific observations"},{"code":"SC.912.L.14.1","desc":"Life Science: Describe the scientific theory of cells"},{"code":"SC.912.P.8.1","desc":"Physical Science: Differentiate among the four states of matter"}],
        },
        "ss_standards": {
            "K-2": [{"code":"SS.K.A.1.1","desc":"History: Develop an understanding of self and how you relate to others"},{"code":"SS.1.G.1.1","desc":"Geography: Use maps, globes, and other geographic tools"}],
            "3-5": [{"code":"SS.3.A.1.1","desc":"History: Explain how groups, events and developments are significant to the emergence of the US"},{"code":"SS.4.G.1.1","desc":"Geography: Identify physical features of Florida"}],
            "6-8": [{"code":"SS.6.W.1.1","desc":"World History: Use timelines to identify chronological order of historical events"},{"code":"SS.7.C.1.1","desc":"Civics: Recognize how Enlightenment ideas influenced American Revolution"},{"code":"SS.8.A.1.1","desc":"American History: Provide supporting details for an answer from text or interview"}],
            "9-12": [{"code":"SS.912.A.1.1","desc":"American History: Describe the importance of historiography in understanding history"},{"code":"SS.912.C.1.1","desc":"Civics: Identify the origins and purposes of law, rules, and government"},{"code":"SS.912.E.1.1","desc":"Economics: Describe how consumers and producers interact"}],
        },
        "assessment_system_prompt": "F.A.S.T. assessment specialist aligned to Florida's B.E.S.T. Standards.",
        "assessment_standard_label": "Standard (e.g. ELA.8.R.1.1 or MA.6.AR.1.1)",
        "assessment_example_placeholder": "e.g. ELA.8.R.1.1 or MA.6.AR.1.1 or leave blank",
        "assessment_result_label": "F.A.S.T. Practice Questions",
    },
    "GA": {
        "name": "Georgia",
        "standards_name": "Georgia Standards of Excellence (GSE)",
        "assessment_name": "Georgia Milestones",
        "standards_body": "Georgia Department of Education",
        "edu_url": "https://www.gadoe.org/",
        "focus_prompt": "You are supporting Georgia K-12 educators. Align all content to Georgia Standards of Excellence (GSE) and Georgia Milestones assessments. Use GSE codes like ELAGSE5RL1 for ELA, MGSE6.EE.1 for Math.",
        "fallback": "[LESSON] GSE-aligned lessons use ELAGSE and MGSE codes for standards-based objectives.\n[ELA] Georgia ELA standards use ELAGSE codes such as ELAGSE8RL1.\n[MATH] Georgia Math standards use MGSE codes such as MGSE6.EE.1.\n[SCIENCE] Georgia Science standards use codes like SKP1 (K), S3L1 (3rd), S7L1 (6-8).",
        "ela_standards": {
            "K-2": [{"code":"ELAGSEKCRF3","desc":"Reading Foundational: Apply phonics and word analysis skills"},{"code":"ELAGSE1RL1","desc":"Reading Literature: Ask and answer questions about key details"},{"code":"ELAGSE2W3","desc":"Writing: Write narratives with details"}],
            "3-5": [{"code":"ELAGSE3RL1","desc":"Reading Literature: Ask and answer questions about text"},{"code":"ELAGSE4RL3","desc":"Reading Literature: Describe character, setting, and events"},{"code":"ELAGSE5RL1","desc":"Reading Literature: Quote accurately from text"}],
            "6-8": [{"code":"ELAGSE6RL1","desc":"Reading Literature: Cite textual evidence to support analysis"},{"code":"ELAGSE7RL1","desc":"Reading Literature: Cite several pieces of evidence"},{"code":"ELAGSE8RL1","desc":"Reading Literature: Cite strongest textual evidence"}],
            "9-12": [{"code":"ELAGSERG1","desc":"Reading Literature: Cite strong and thorough textual evidence"},{"code":"ELAGSERGW1","desc":"Writing: Write arguments to support claims"},{"code":"ELAGSERGW3","desc":"Writing: Write narratives with narrative techniques"}],
        },
        "math_standards": {
            "K-2": [{"code":"MGSEK.CC.1","desc":"Counting: Count to 100 by ones and tens"},{"code":"MGSE1.OA.1","desc":"Operations: Use addition and subtraction within 20"}],
            "3-5": [{"code":"MGSE3.OA.1","desc":"Operations: Interpret products of whole numbers"},{"code":"MGSE4.NF.1","desc":"Fractions: Explain equivalent fractions using models"},{"code":"MGSE5.NBT.1","desc":"Number: Digit in one place is 10 times the digit to its right"}],
            "6-8": [{"code":"MGSE6.EE.1","desc":"Expressions: Write and evaluate numerical expressions"},{"code":"MGSE7.NS.1","desc":"Number System: Add and subtract rational numbers"},{"code":"MGSE8.EE.5","desc":"Expressions: Graph proportional relationships"}],
            "9-12": [{"code":"MGSE9-12.A.CED.1","desc":"Algebra: Create equations and inequalities to solve problems"},{"code":"MGSE9-12.F.IF.1","desc":"Functions: Understand that a function assigns exactly one output"},{"code":"MGSE9-12.S.ID.1","desc":"Statistics: Represent data with plots"}],
        },
        "science_label": "Science (Georgia Standards)",
        "science_standards": {
            "K-2": [{"code":"SKP1","desc":"Physical Science: Describe, compare and sort objects"},{"code":"SKE1","desc":"Earth Science: Describe the physical attributes of stars"}],
            "3-5": [{"code":"S3L1","desc":"Life Science: Investigate the habitats of different organisms"},{"code":"S4P3","desc":"Physical Science: Demonstrate the relationship between electricity and magnetism"}],
            "6-8": [{"code":"S7L1","desc":"Life Science: Investigate and understand structure and function of cells"},{"code":"S8P1","desc":"Physical Science: Investigate and understand structure and properties of matter"}],
            "9-12": [{"code":"SB1","desc":"Biology: Obtain, evaluate and communicate information to analyze genetic information"},{"code":"SC1","desc":"Chemistry: Obtain, evaluate and communicate information about atomic structure"}],
        },
        "ss_standards": {
            "K-2": [{"code":"SSKE1","desc":"History: Explain the importance of celebrating and practicing democracy"},{"code":"SKG1","desc":"Geography: Describe the physical attributes of maps and globes"}],
            "3-5": [{"code":"SS3H1","desc":"History: Describe early settlements of American colonies"},{"code":"SS4G1","desc":"Geography: Describe physical and human characteristics of Georgia"}],
            "6-8": [{"code":"SS6H1","desc":"World History: Describe the early civilizations of Africa"},{"code":"SS8H1","desc":"Georgia History: Evaluate the impact of European exploration on Georgia"}],
            "9-12": [{"code":"SSUSH1","desc":"US History: Describe the development of ideas that led to the American Revolution"},{"code":"SSECO1","desc":"Economics: Explain how economic decisions affect the well-being of individuals, businesses, and society"}],
        },
        "assessment_system_prompt": "Georgia Milestones assessment specialist aligned to Georgia Standards of Excellence (GSE).",
        "assessment_standard_label": "Standard (e.g. ELAGSE8RL1 or MGSE6.EE.1)",
        "assessment_example_placeholder": "e.g. ELAGSE8RL1 or MGSE6.EE.1 or leave blank",
        "assessment_result_label": "Georgia Milestones Practice Questions",
    },
    "HI": _ccss_state("Hawaii", "Hawaii Common Core Standards", "Smarter Balanced",
        "https://www.hawaiipublicschools.org/",
        "You are supporting Hawaii K-12 educators. Align content to Hawaii Common Core Standards and Smarter Balanced assessments. Use CCSS-aligned codes for ELA and Math.",
        "Smarter Balanced assessment specialist aligned to Hawaii Common Core Standards.",
        "Smarter Balanced Practice Questions"),
    "ID": _ccss_state("Idaho", "Idaho Content Standards", "ISAT",
        "https://www.sde.idaho.gov/",
        "You are supporting Idaho K-12 educators. Align content to Idaho Content Standards and ISAT assessments. Use CCSS-aligned codes for ELA and Math.",
        "ISAT assessment specialist aligned to Idaho Content Standards.",
        "ISAT Practice Questions"),
    "IL": _ccss_state("Illinois", "Illinois Learning Standards", "IAR",
        "https://www.isbe.net/",
        "You are supporting Illinois K-12 educators. Align content to Illinois Learning Standards and IAR assessments. Use CCSS-aligned codes for ELA and Math.",
        "IAR assessment specialist aligned to Illinois Learning Standards.",
        "IAR Practice Questions"),
    "IN": _ccss_state("Indiana", "Indiana Academic Standards", "ILEARN",
        "https://www.in.gov/doe/",
        "You are supporting Indiana K-12 educators. Align content to Indiana Academic Standards and ILEARN assessments. Use CCSS-aligned codes for ELA and Math.",
        "ILEARN assessment specialist aligned to Indiana Academic Standards.",
        "ILEARN Practice Questions"),
    "IA": _ccss_state("Iowa", "Iowa Core Standards", "ISASP",
        "https://educateiowa.gov/",
        "You are supporting Iowa K-12 educators. Align content to Iowa Core Standards and ISASP assessments. Use CCSS-aligned codes for ELA and Math.",
        "ISASP assessment specialist aligned to Iowa Core Standards.",
        "ISASP Practice Questions"),
    "KS": _ccss_state("Kansas", "Kansas College and Career Ready Standards", "KAP",
        "https://www.ksde.org/",
        "You are supporting Kansas K-12 educators. Align content to Kansas College and Career Ready Standards and KAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "KAP assessment specialist aligned to Kansas College and Career Ready Standards.",
        "KAP Practice Questions"),
    "KY": _ccss_state("Kentucky", "Kentucky Academic Standards", "KSA",
        "https://education.ky.gov/",
        "You are supporting Kentucky K-12 educators. Align content to Kentucky Academic Standards and KSA assessments. Use CCSS-aligned codes for ELA and Math.",
        "KSA assessment specialist aligned to Kentucky Academic Standards.",
        "KSA Practice Questions"),
    "LA": _ccss_state("Louisiana", "Louisiana Student Standards", "LEAP 2025",
        "https://www.louisianabelieves.com/",
        "You are supporting Louisiana K-12 educators. Align content to Louisiana Student Standards and LEAP 2025 assessments. Use CCSS-aligned codes for ELA and Math.",
        "LEAP 2025 assessment specialist aligned to Louisiana Student Standards.",
        "LEAP 2025 Practice Questions"),
    "ME": _ccss_state("Maine", "Maine Learning Results", "MEA",
        "https://www.maine.gov/doe/",
        "You are supporting Maine K-12 educators. Align content to Maine Learning Results and MEA assessments. Use CCSS-aligned codes for ELA and Math.",
        "MEA assessment specialist aligned to Maine Learning Results.",
        "MEA Practice Questions"),
    "MD": _ccss_state("Maryland", "Maryland College and Career Ready Standards", "MCAP",
        "https://marylandpublicschools.org/",
        "You are supporting Maryland K-12 educators. Align content to Maryland College and Career Ready Standards and MCAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "MCAP assessment specialist aligned to Maryland College and Career Ready Standards.",
        "MCAP Practice Questions"),
    "MA": _ccss_state("Massachusetts", "Massachusetts Curriculum Frameworks", "MCAS",
        "https://www.doe.mass.edu/",
        "You are supporting Massachusetts K-12 educators. Align content to Massachusetts Curriculum Frameworks and MCAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "MCAS assessment specialist aligned to Massachusetts Curriculum Frameworks.",
        "MCAS Practice Questions"),
    "MI": _ccss_state("Michigan", "Michigan K-12 Standards", "M-STEP",
        "https://www.michigan.gov/mde",
        "You are supporting Michigan K-12 educators. Align content to Michigan K-12 Standards and M-STEP assessments. Use CCSS-aligned codes for ELA and Math.",
        "M-STEP assessment specialist aligned to Michigan K-12 Standards.",
        "M-STEP Practice Questions"),
    "MN": _ccss_state("Minnesota", "Minnesota Academic Standards", "MCA",
        "https://education.mn.gov/",
        "You are supporting Minnesota K-12 educators. Align content to Minnesota Academic Standards and MCA assessments. Use CCSS-aligned codes for ELA and Math.",
        "MCA assessment specialist aligned to Minnesota Academic Standards.",
        "MCA Practice Questions"),
    "MS": _ccss_state("Mississippi", "MS College- and Career-Readiness Standards", "MAAP",
        "https://www.mdek12.org/",
        "You are supporting Mississippi K-12 educators. Align content to MS College- and Career-Readiness Standards and MAAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "MAAP assessment specialist aligned to MS College- and Career-Readiness Standards.",
        "MAAP Practice Questions"),
    "MO": _ccss_state("Missouri", "Missouri Learning Standards", "MAP",
        "https://dese.mo.gov/",
        "You are supporting Missouri K-12 educators. Align content to Missouri Learning Standards and MAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "MAP assessment specialist aligned to Missouri Learning Standards.",
        "MAP Practice Questions"),
    "MT": _ccss_state("Montana", "Montana Content Standards", "Montana Assessment",
        "https://opi.mt.gov/",
        "You are supporting Montana K-12 educators. Align content to Montana Content Standards and Montana Assessment. Use CCSS-aligned codes for ELA and Math. Incorporate Montana's unique cultural and geographic contexts.",
        "Montana Assessment specialist aligned to Montana Content Standards.",
        "Montana Assessment Practice Questions"),
    "NE": _ccss_state("Nebraska", "Nebraska Content Area Standards", "NSCAS",
        "https://www.education.ne.gov/",
        "You are supporting Nebraska K-12 educators. Align content to Nebraska Content Area Standards and NSCAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "NSCAS assessment specialist aligned to Nebraska Content Area Standards.",
        "NSCAS Practice Questions"),
    "NV": _ccss_state("Nevada", "Nevada Academic Content Standards", "Nevada CRT/ACT",
        "https://doe.nv.gov/",
        "You are supporting Nevada K-12 educators. Align content to Nevada Academic Content Standards and Nevada CRT/ACT assessments. Use CCSS-aligned codes for ELA and Math.",
        "Nevada CRT assessment specialist aligned to Nevada Academic Content Standards.",
        "Nevada CRT Practice Questions"),
    "NH": _ccss_state("New Hampshire", "NH College and Career Ready Standards", "NH SAS",
        "https://www.education.nh.gov/",
        "You are supporting New Hampshire K-12 educators. Align content to NH College and Career Ready Standards and NH SAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "NH SAS assessment specialist aligned to NH College and Career Ready Standards.",
        "NH SAS Practice Questions"),
    "NJ": {
        "name": "New Jersey",
        "standards_name": "NJ Student Learning Standards (NJSLS)",
        "assessment_name": "NJSLA",
        "standards_body": "New Jersey Department of Education",
        "edu_url": "https://www.nj.gov/education/",
        "focus_prompt": "You are supporting New Jersey K-12 educators. Align all content to NJ Student Learning Standards (NJSLS) and NJSLA assessments. Use standard codes like 6.1.8.CivicsPD.1 for Social Studies and RL.8.1/CCSS codes for ELA and Math.",
        "fallback": "[LESSON] NJSLS-aligned lessons use CCSS codes for ELA/Math and NJ-specific codes for Social Studies (e.g., 6.1.8.CivicsPD.1).\n[ELA] NJ ELA standards are CCSS-based with codes like RL.8.1 and W.7.1.\n[MATH] NJ Math standards are CCSS-based with codes like 8.EE.B.5.\n[SOCIAL STUDIES] NJ Social Studies uses codes like 6.1.8.CivicsPD.1.\n[SCIENCE] NJ Science uses NGSS codes.",
        "ela_standards": _CCSS_ELA_STDS,
        "math_standards": _CCSS_MATH_STDS,
        "science_label": "Science (NGSS)",
        "science_standards": _NGSS_SCI_STDS,
        "ss_standards": {
            "K-2": [{"code":"6.1.2.CivicsPD.1","desc":"Civics: Explain how classroom rules help everyone"},{"code":"6.1.2.GeoPP.1","desc":"Geography: Describe how location affects daily life"}],
            "3-5": [{"code":"6.1.5.CivicsPD.1","desc":"Civics: Explain how democratic processes work"},{"code":"6.1.5.HistoryCC.3","desc":"History: Sequence key events of colonial America"}],
            "6-8": [{"code":"6.1.8.CivicsPD.1","desc":"Civics: Analyze how democratic ideals are reflected in the Constitution"},{"code":"6.1.8.HistoryCC.3","desc":"History: Analyze causes and effects of the Civil War"}],
            "9-12": [{"code":"6.1.12.CivicsPD.4","desc":"Civics: Analyze how Constitutional amendments expanded civil rights"},{"code":"6.1.12.EconGE.1","desc":"Economics: Evaluate causes and effects of the Great Depression"}],
        },
        "assessment_system_prompt": "NJSLA assessment specialist aligned to NJ Student Learning Standards.",
        "assessment_standard_label": "Standard (e.g. RL.8.1 or 6.1.8.CivicsPD.1)",
        "assessment_example_placeholder": "e.g. RL.8.1 or 6.1.8.CivicsPD.1",
        "assessment_result_label": "NJSLA Practice Questions",
    },
    "NM": _ccss_state("New Mexico", "NM Common Core Standards", "NM MSSA",
        "https://webnew.ped.state.nm.us/",
        "You are supporting New Mexico K-12 educators. Align content to NM Common Core Standards and NM MSSA assessments. Use CCSS-aligned codes for ELA and Math.",
        "NM MSSA assessment specialist aligned to New Mexico Common Core Standards.",
        "NM MSSA Practice Questions"),
    "NY": {
        "name": "New York",
        "standards_name": "NY Next Generation Learning Standards",
        "assessment_name": "NY State/Regents",
        "standards_body": "New York State Education Department",
        "edu_url": "https://www.nysed.gov/",
        "focus_prompt": "You are supporting New York K-12 educators. Align all content to New York Next Generation Learning Standards and NY State Assessments/Regents exams. Use NY-specific codes like KR1, 3R1 for ELA and NY-6.RP.1 for Math.",
        "fallback": "[LESSON] NY Next Generation Learning Standards align to Regents examinations and NY State tests.\n[ELA] NY ELA standards use codes like KR1, 3R1, 8R1.\n[MATH] NY Math standards use codes like NY-6.RP.1, NY-8.EE.B.5.\n[SCIENCE] NY Science uses NGSS-based NYSSLS standards.",
        "ela_standards": {
            "K-2": [{"code":"KR1","desc":"Reading: With prompting, ask/answer questions about key details"},{"code":"1R1","desc":"Reading: Ask and answer questions about key details"},{"code":"2W3","desc":"Writing: Write narratives with details"}],
            "3-5": [{"code":"3R1","desc":"Reading: Ask and answer questions about text"},{"code":"4R3","desc":"Reading: Describe characters, settings and events"},{"code":"5R1","desc":"Reading: Quote accurately from text when explaining"}],
            "6-8": [{"code":"6R1","desc":"Reading: Cite textual evidence to support analysis"},{"code":"7R1","desc":"Reading: Cite several pieces of textual evidence"},{"code":"8R1","desc":"Reading: Cite strongest evidence to support inference"}],
            "9-12": [{"code":"9R1","desc":"Reading: Cite strong evidence to support analysis"},{"code":"9-10W1","desc":"Writing: Write arguments with precise claims"},{"code":"11-12W3","desc":"Writing: Write narratives with pacing and description"}],
        },
        "math_standards": {
            "K-2": [{"code":"NY-K.CC.1","desc":"Counting: Count to 100 by ones and tens"},{"code":"NY-1.OA.1","desc":"Operations: Use addition and subtraction within 20"}],
            "3-5": [{"code":"NY-3.OA.1","desc":"Operations: Interpret products of whole numbers"},{"code":"NY-4.NF.1","desc":"Fractions: Explain equivalent fractions"},{"code":"NY-5.NBT.1","desc":"Number: Digit in one place is 10 times digit to its right"}],
            "6-8": [{"code":"NY-6.RP.1","desc":"Ratios: Understand concept of ratio and use ratio language"},{"code":"NY-7.NS.1","desc":"Number System: Add and subtract rational numbers"},{"code":"NY-8.EE.5","desc":"Expressions: Graph proportional relationships, interpreting slope"}],
            "9-12": [{"code":"NY-A.CED.1","desc":"Algebra: Create equations and inequalities to solve problems"},{"code":"NY-F.IF.1","desc":"Functions: Understand that a function assigns exactly one output"},{"code":"NY-S.ID.1","desc":"Statistics: Represent data with plots"}],
        },
        "science_label": "Science (NYSSLS)",
        "science_standards": _NGSS_SCI_STDS,
        "assessment_system_prompt": "NY State Regents and Next Generation assessment specialist aligned to New York's learning standards.",
        "assessment_standard_label": "Standard (e.g. 6R1 or NY-6.RP.1)",
        "assessment_example_placeholder": "e.g. 6R1 or NY-6.RP.1 or leave blank",
        "assessment_result_label": "NY State/Regents Practice Questions",
    },
    "NC": _ccss_state("North Carolina", "NC Standard Course of Study", "NC EOG/EOC",
        "https://www.dpi.nc.gov/",
        "You are supporting North Carolina K-12 educators. Align content to NC Standard Course of Study and NC EOG/EOC assessments. Use CCSS-aligned codes for ELA and Math.",
        "NC EOG/EOC assessment specialist aligned to NC Standard Course of Study.",
        "NC EOG/EOC Practice Questions"),
    "ND": _ccss_state("North Dakota", "ND Academic Content Standards", "NDSA",
        "https://www.nd.gov/dpi/",
        "You are supporting North Dakota K-12 educators. Align content to ND Academic Content Standards and NDSA assessments. Use CCSS-aligned codes for ELA and Math.",
        "NDSA assessment specialist aligned to ND Academic Content Standards.",
        "NDSA Practice Questions"),
    "OH": _ccss_state("Ohio", "Ohio's Learning Standards", "Ohio's State Tests",
        "https://education.ohio.gov/",
        "You are supporting Ohio K-12 educators. Align content to Ohio's Learning Standards and Ohio's State Tests. Use CCSS-aligned codes for ELA and Math.",
        "Ohio State Tests specialist aligned to Ohio's Learning Standards.",
        "Ohio State Tests Practice Questions"),
    "OK": _ccss_state("Oklahoma", "Oklahoma Academic Standards", "OSTP",
        "https://sde.ok.gov/",
        "You are supporting Oklahoma K-12 educators. Align content to Oklahoma Academic Standards and OSTP assessments. Use CCSS-aligned codes for ELA and Math.",
        "OSTP assessment specialist aligned to Oklahoma Academic Standards.",
        "OSTP Practice Questions"),
    "OR": _ccss_state("Oregon", "Oregon State Standards", "OSAS",
        "https://www.oregon.gov/ode/",
        "You are supporting Oregon K-12 educators. Align content to Oregon State Standards and OSAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "OSAS assessment specialist aligned to Oregon State Standards.",
        "OSAS Practice Questions"),
    "PA": {
        "name": "Pennsylvania",
        "standards_name": "Pennsylvania Core Standards",
        "assessment_name": "PSSA/Keystone",
        "standards_body": "Pennsylvania Department of Education",
        "edu_url": "https://www.education.pa.gov/",
        "focus_prompt": "You are supporting Pennsylvania K-12 educators. Align all content to Pennsylvania Core Standards and PSSA/Keystone assessments. Use PA Core codes like CC.1.2.3.A for ELA and CC.2.1.4.B.2 for Math.",
        "fallback": "[LESSON] PA Core Standards use codes like CC.1.2.3.A for ELA and CC.2.1.4.B.2 for Math.\n[ELA] Pennsylvania ELA standards use PA Core codes like CC.1.3.6.A.\n[MATH] Pennsylvania Math standards use PA Core codes like CC.2.2.6.B.1.\n[SCIENCE] Pennsylvania Science uses NGSS-aligned standards.",
        "ela_standards": {
            "K-2": [{"code":"CC.1.1.K.A","desc":"Reading: Demonstrate understanding of spoken words, syllables, and sounds"},{"code":"CC.1.2.1.A","desc":"Reading Informational: Identify main idea and key details"},{"code":"CC.1.4.K.A","desc":"Writing: Use drawing, dictating, and writing to narrate events"}],
            "3-5": [{"code":"CC.1.2.3.A","desc":"Reading Informational: Determine main idea and explain details"},{"code":"CC.1.3.4.A","desc":"Reading Literature: Determine a theme of a story"},{"code":"CC.1.4.3.A","desc":"Writing: Write opinion pieces on topics"}],
            "6-8": [{"code":"CC.1.3.6.A","desc":"Reading Literature: Determine theme or central idea"},{"code":"CC.1.2.6.A","desc":"Reading Informational: Determine central idea in text"},{"code":"CC.1.4.6.A","desc":"Writing: Write arguments to support claims"}],
            "9-12": [{"code":"CC.1.3.11.A","desc":"Reading Literature: Determine meaning of text and analyze author's choices"},{"code":"CC.1.2.11.A","desc":"Reading Informational: Determine and analyze author's central idea"},{"code":"CC.1.4.11.A","desc":"Writing: Write arguments with evidence and reasoning"}],
        },
        "math_standards": {
            "K-2": [{"code":"CC.2.1.K.B.1","desc":"Number Sense: Use place value to compose and decompose numbers"},{"code":"CC.2.2.1.A.1","desc":"Operations: Represent and solve problems with addition and subtraction"}],
            "3-5": [{"code":"CC.2.1.3.B.1","desc":"Number Sense: Apply place value to round and compare numbers"},{"code":"CC.2.4.3.A.1","desc":"Measurement: Solve problems involving measurement of time intervals"}],
            "6-8": [{"code":"CC.2.2.6.B.1","desc":"Expressions: Apply properties to create equivalent expressions"},{"code":"CC.2.1.7.E.1","desc":"Number System: Apply properties to add and subtract rational numbers"}],
            "9-12": [{"code":"CC.2.2.HS.D.1","desc":"Algebra: Interpret the structure of expressions"},{"code":"CC.2.2.HS.C.1","desc":"Functions: Interpret functions in terms of context"},{"code":"CC.2.4.HS.B.1","desc":"Statistics: Summarize, represent, and interpret data"}],
        },
        "science_label": "Science (PA Standards)",
        "science_standards": {
            "K-2": [{"code":"3.1.K.A1","desc":"Biological Sciences: Identify living and non-living things"},{"code":"3.3.K.A1","desc":"Earth and Space Science: Identify basic properties of earth materials"}],
            "3-5": [{"code":"3.1.3.A1","desc":"Biological Sciences: Describe the life cycles of organisms"},{"code":"3.3.3.A1","desc":"Earth and Space Science: Identify basic properties of minerals, rocks and soils"}],
            "6-8": [{"code":"3.1.6.A1","desc":"Biological Sciences: Explain the importance of cell structure and function"},{"code":"3.3.6.A1","desc":"Earth and Space Science: Describe the layers of the Earth"}],
            "9-12": [{"code":"3.1.12.A1","desc":"Biological Sciences: Analyze the molecular basis of heredity"},{"code":"3.3.12.A1","desc":"Earth and Space Science: Explain the role of the sun in Earth's energy budget"}],
        },
        "ss_standards": {
            "K-2": [{"code":"5.1.K.A","desc":"Civics: Identify rules and laws in the community"},{"code":"7.1.K.A","desc":"Geography: Identify geographic tools"}],
            "3-5": [{"code":"5.1.3.A","desc":"Civics: Explain the purposes of rules and laws"},{"code":"8.1.3.A","desc":"History: Identify contributions of historical figures"}],
            "6-8": [{"code":"5.1.6.A","desc":"Civics: Explain the principle of limited government"},{"code":"8.2.6.A","desc":"History: Explain how historical events develop over time"}],
            "9-12": [{"code":"5.1.9.A","desc":"Civics: Analyze the role of limited government in protecting citizens"},{"code":"8.1.9.A","desc":"History: Evaluate the significance of major historical documents"}],
        },
        "assessment_system_prompt": "PSSA and Keystone Exam specialist aligned to Pennsylvania Core Standards.",
        "assessment_standard_label": "Standard (e.g. CC.1.2.3.A or CC.2.1.4.B.2)",
        "assessment_example_placeholder": "e.g. CC.1.2.3.A or CC.2.1.4.B.2 or leave blank",
        "assessment_result_label": "PSSA/Keystone Practice Questions",
    },
    "RI": _ccss_state("Rhode Island", "Rhode Island State Standards", "RICAS",
        "https://www.ride.ri.gov/",
        "You are supporting Rhode Island K-12 educators. Align content to Rhode Island State Standards and RICAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "RICAS assessment specialist aligned to Rhode Island State Standards.",
        "RICAS Practice Questions"),
    "SC": _ccss_state("South Carolina", "SC College- and Career-Ready Standards", "SC READY",
        "https://ed.sc.gov/",
        "You are supporting South Carolina K-12 educators. Align content to SC College- and Career-Ready Standards and SC READY assessments. Use CCSS-aligned codes for ELA and Math.",
        "SC READY assessment specialist aligned to SC College- and Career-Ready Standards.",
        "SC READY Practice Questions"),
    "SD": _ccss_state("South Dakota", "SD Content Standards", "SD STEP",
        "https://doe.sd.gov/",
        "You are supporting South Dakota K-12 educators. Align content to SD Content Standards and SD STEP assessments. Use CCSS-aligned codes for ELA and Math.",
        "SD STEP assessment specialist aligned to South Dakota Content Standards.",
        "SD STEP Practice Questions"),
    "TN": _ccss_state("Tennessee", "Tennessee Academic Standards", "TCAP",
        "https://www.tn.gov/education.html",
        "You are supporting Tennessee K-12 educators. Align content to Tennessee Academic Standards and TCAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "TCAP assessment specialist aligned to Tennessee Academic Standards.",
        "TCAP Practice Questions"),
    "TX": {
        "name": "Texas",
        "standards_name": "Texas Essential Knowledge and Skills (TEKS)",
        "assessment_name": "STAAR",
        "standards_body": "Texas Education Agency",
        "edu_url": "https://tea.texas.gov/",
        "focus_prompt": "You are supporting Texas K-12 educators. Align all content to Texas Essential Knowledge and Skills (TEKS) and STAAR assessments. Use TEKS codes like 2.6A for Math, ELA.3.4A for English Language Arts, and SCI.7.5A for Science.",
        "fallback": "[LESSON] Texas TEKS use codes like ELA.3.4A for ELA, 2.6A for Math, and SCI.7.5A for Science.\n[ELA] Texas ELA TEKS use codes like ELA.3.4A, ELA.8.5A for Reading and Writing.\n[MATH] Texas Math TEKS use codes like K.2A, 3.4A, 8.5A, A.2A.\n[SCIENCE] Texas Science TEKS use codes like SCI.K.2A, SCI.7.5A.",
        "ela_standards": {
            "K-2": [{"code":"ELA.K.1A","desc":"Reading: Identify and describe the topic of a text"},{"code":"ELA.1.5A","desc":"Reading: Use comprehension skills to analyze literary text"},{"code":"ELA.2.5A","desc":"Reading: Identify elements of literary text including setting and plot"}],
            "3-5": [{"code":"ELA.3.4A","desc":"Reading: Use comprehension skills to analyze literary text"},{"code":"ELA.4.4A","desc":"Reading: Analyze literary text including theme and character"},{"code":"ELA.5.4A","desc":"Reading: Analyze literary text to make complex inferences"}],
            "6-8": [{"code":"ELA.6.4A","desc":"Reading: Analyze literary text to make complex inferences"},{"code":"ELA.7.5A","desc":"Reading: Analyze how figurative language contributes to meaning"},{"code":"ELA.8.5A","desc":"Reading: Analyze how tone is created through word choice"}],
            "9-12": [{"code":"ELA.E1.4A","desc":"Reading: Analyze literary text including theme, plot, and characterization"},{"code":"ELA.E2.4A","desc":"Reading: Analyze how literary elements contribute to meaning"},{"code":"ELA.E1.5B","desc":"Writing: Compose literary texts using literary devices"}],
        },
        "math_standards": {
            "K-2": [{"code":"K.2A","desc":"Number: Count forward and backward to 100"},{"code":"1.3A","desc":"Number: Use concrete models to represent addition and subtraction"},{"code":"2.6A","desc":"Geometry: Classify and sort polygons with 12 or fewer sides"}],
            "3-5": [{"code":"3.4A","desc":"Number: Solve with fluency one-step problems using multiplication and division"},{"code":"4.4A","desc":"Number: Add and subtract whole numbers and decimals"},{"code":"5.3A","desc":"Number: Estimate to determine solutions to math problems"}],
            "6-8": [{"code":"6.4A","desc":"Proportionality: Compare two rules verbally, numerically, graphically"},{"code":"7.4A","desc":"Proportionality: Represent constant rates of change in mathematical and real-world problems"},{"code":"8.5A","desc":"Proportionality: Represent linear proportional situations with tables, graphs, and equations"}],
            "9-12": [{"code":"A.2A","desc":"Algebra 1: Determine the domain and range of functions"},{"code":"A.7A","desc":"Algebra 1: Graph quadratic functions on the coordinate plane"},{"code":"G.2A","desc":"Geometry: Determine coordinates of a point that is a given fraction"}],
        },
        "science_label": "Science (TEKS)",
        "science_standards": {
            "K-2": [{"code":"SCI.K.2A","desc":"Scientific Process: Ask questions about organisms and events in nature"},{"code":"SCI.1.6A","desc":"Force and Motion: Identify how different forces such as gravity and friction act on objects"}],
            "3-5": [{"code":"SCI.3.3A","desc":"Matter: Measure, test and record physical properties of matter"},{"code":"SCI.4.3A","desc":"Matter: Describe and predict the changes caused by heat and light"},{"code":"SCI.5.5A","desc":"Matter: Classify matter based on physical and chemical properties"}],
            "6-8": [{"code":"SCI.6.4A","desc":"Matter: Differentiate between elements and compounds"},{"code":"SCI.7.5A","desc":"Matter: Recognize that chemical formulas are used to identify substances"},{"code":"SCI.8.6A","desc":"Force/Motion: Calculate how unbalanced forces change the speed of an object"}],
            "9-12": [{"code":"SCI.B.4A","desc":"Biology: Explain and analyze how the process of natural selection occurs"},{"code":"SCI.C.5A","desc":"Chemistry: Describe the relationship of the properties of elements"},{"code":"SCI.P.4A","desc":"Physics: Describe and analyze motion in one dimension"}],
        },
        "ss_standards": {
            "K-2": [{"code":"SS.K.1A","desc":"Citizenship: Identify characteristics of good citizenship"},{"code":"SS.2.5A","desc":"Geography: Describe the physical characteristics of Texas"}],
            "3-5": [{"code":"SS.3.1A","desc":"History: Identify individuals who have contributed to Texas and U.S. history"},{"code":"SS.5.1A","desc":"History: Identify significant individuals who shaped U.S. history"}],
            "6-8": [{"code":"SS.6.1A","desc":"World History: Describe the influence of individual and group actions"},{"code":"SS.8.1A","desc":"US History: Explain causes of the Texas Revolution"}],
            "9-12": [{"code":"SS.10.1A","desc":"World History: Identify significant individuals and events in world history"},{"code":"SS.11.1A","desc":"US History: Trace the development of American democracy"}],
        },
        "assessment_system_prompt": "STAAR assessment specialist aligned to Texas Essential Knowledge and Skills (TEKS).",
        "assessment_standard_label": "Standard (e.g. 8.5A or ELA.8.5A)",
        "assessment_example_placeholder": "e.g. 8.5A or ELA.8.5A or leave blank",
        "assessment_result_label": "STAAR Practice Questions",
    },
    "UT": _ccss_state("Utah", "Utah Core Standards", "RISE",
        "https://www.schools.utah.gov/",
        "You are supporting Utah K-12 educators. Align content to Utah Core Standards and RISE assessments. Use CCSS-aligned codes for ELA and Math.",
        "RISE assessment specialist aligned to Utah Core Standards.",
        "RISE Practice Questions"),
    "VT": _ccss_state("Vermont", "Vermont Common Core Standards", "VTCAP",
        "https://education.vermont.gov/",
        "You are supporting Vermont K-12 educators. Align content to Vermont Common Core Standards and VTCAP assessments. Use CCSS-aligned codes for ELA and Math.",
        "VTCAP assessment specialist aligned to Vermont Common Core Standards.",
        "VTCAP Practice Questions"),
    "VA": {
        "name": "Virginia",
        "standards_name": "Virginia Standards of Learning (SOL)",
        "assessment_name": "SOL Tests",
        "standards_body": "Virginia Department of Education",
        "edu_url": "https://www.doe.virginia.gov/",
        "focus_prompt": "You are supporting Virginia K-12 educators. Align all content to Virginia Standards of Learning (SOL) and SOL Tests. Use SOL codes like 3.5a, SOL 5.4, ELA.6.5, and A.2.",
        "fallback": "[LESSON] Virginia SOL-aligned lessons use codes like ELA.6.5 for ELA and SOL 3.3 for Math.\n[ELA] Virginia ELA SOL use codes like ELA.K.1, ELA.3.5, ELA.6.5.\n[MATH] Virginia Math SOL use codes like K.1, 3.3, 6.7, A.2.\n[SCIENCE] Virginia Science SOL use codes like K.1, 3.1, LS.1.",
        "ela_standards": {
            "K-2": [{"code":"ELA.K.1","desc":"Reading: Develop oral language and listening skills"},{"code":"ELA.1.7","desc":"Reading: Understand the meaning of common words in text"},{"code":"ELA.2.11","desc":"Writing: Write stories, letters, or simple explanations"}],
            "3-5": [{"code":"ELA.3.5","desc":"Reading: Read and demonstrate comprehension of fictional text"},{"code":"ELA.4.5","desc":"Reading: Read and demonstrate comprehension of fictional text"},{"code":"ELA.5.6","desc":"Reading: Read and demonstrate comprehension of nonfiction text"}],
            "6-8": [{"code":"ELA.6.5","desc":"Reading: Read and demonstrate comprehension of fictional text"},{"code":"ELA.7.5","desc":"Reading: Read and demonstrate comprehension of fictional text"},{"code":"ELA.8.5","desc":"Reading: Read and demonstrate comprehension of fictional text"}],
            "9-12": [{"code":"ELA.9.5","desc":"Reading: Read and analyze fictional text"},{"code":"ELA.10.5","desc":"Reading: Read and analyze fictional and nonfiction text"},{"code":"ELA.11.5","desc":"Reading: Read, interpret, analyze, and evaluate nonfiction text"}],
        },
        "math_standards": {
            "K-2": [{"code":"K.1","desc":"Number: Count forward to 110 and backward from 30"},{"code":"1.1","desc":"Number: Count forward by ones to 110 and backward from 30"},{"code":"2.1","desc":"Number: Read, write, and identify place value"}],
            "3-5": [{"code":"3.3","desc":"Number: Read and write six-digit numerals and identify place value"},{"code":"4.4","desc":"Number: Estimate sums, differences, products and quotients"},{"code":"5.4","desc":"Fractions: Create and solve problems involving addition and subtraction of fractions"}],
            "6-8": [{"code":"6.7","desc":"Equations: Write and solve one-step equations using whole numbers"},{"code":"7.12","desc":"Equations: Represent relationships with linear equations"},{"code":"8.17","desc":"Statistics: Describe and represent relations and functions"}],
            "9-12": [{"code":"A.2","desc":"Algebra: Represent verbal quantitative situations algebraically and evaluate expressions"},{"code":"A.4","desc":"Algebra: Solve multistep linear and quadratic equations"},{"code":"G.3","desc":"Geometry: Use deductive reasoning to construct logical arguments"}],
        },
        "science_label": "Science (Virginia SOL)",
        "science_standards": {
            "K-2": [{"code":"K.1","desc":"Scientific Investigation: Conduct basic experiments and communicate findings"},{"code":"1.1","desc":"Scientific Investigation: Plan and conduct investigations"}],
            "3-5": [{"code":"3.1","desc":"Scientific Investigation: Plan and conduct experiments"},{"code":"4.1","desc":"Scientific Investigation: Demonstrate understanding of scientific reasoning"}],
            "6-8": [{"code":"LS.1","desc":"Life Science: Demonstrate understanding of scientific reasoning"},{"code":"ES.1","desc":"Earth Science: Plan and conduct investigations"}],
            "9-12": [{"code":"CH.1","desc":"Chemistry: Plan and conduct investigations using safe and ethical procedures"},{"code":"BIO.1","desc":"Biology: Plan and conduct investigations"}],
        },
        "ss_standards": {
            "K-2": [{"code":"VS.1","desc":"Virginia Studies: Describe reasons people came to Virginia"},{"code":"K.1","desc":"Social Studies: Recognize that history describes events and people of other times and places"}],
            "3-5": [{"code":"VS.3","desc":"Virginia Studies: Demonstrate understanding of how geography influenced growth of Virginia"},{"code":"USI.3","desc":"US History: Describe colonial America"}],
            "6-8": [{"code":"USII.3","desc":"US History II: Examine the social, economic, and political transformation of the US"}],
            "9-12": [{"code":"GOVT.1","desc":"Government: Demonstrate understanding of principles of government, including limited government"},{"code":"GOVT.4","desc":"Government: Demonstrate understanding of the rights, freedoms and responsibilities of citizens"}],
        },
        "assessment_system_prompt": "Virginia SOL assessment specialist aligned to Standards of Learning.",
        "assessment_standard_label": "Standard (e.g. 3.5a or ELA.6.5)",
        "assessment_example_placeholder": "e.g. 3.5a or ELA.6.5 or leave blank",
        "assessment_result_label": "SOL Practice Questions",
    },
    "WA": _ccss_state("Washington", "Washington State Learning Standards", "SBA/WCAS",
        "https://www.k12.wa.us/",
        "You are supporting Washington State K-12 educators. Align content to Washington State Learning Standards and SBA/WCAS assessments. Use CCSS-aligned codes for ELA and Math.",
        "SBA/WCAS assessment specialist aligned to Washington State Learning Standards.",
        "SBA/WCAS Practice Questions"),
    "WV": _ccss_state("West Virginia", "WV College- and Career-Readiness Standards", "WVGSA",
        "https://wvde.us/",
        "You are supporting West Virginia K-12 educators. Align content to WV College- and Career-Readiness Standards and WVGSA assessments. Use CCSS-aligned codes for ELA and Math.",
        "WVGSA assessment specialist aligned to WV College- and Career-Readiness Standards.",
        "WVGSA Practice Questions"),
    "WI": _ccss_state("Wisconsin", "Wisconsin Academic Standards", "Forward Exam",
        "https://dpi.wi.gov/",
        "You are supporting Wisconsin K-12 educators. Align content to Wisconsin Academic Standards and Forward Exam assessments. Use CCSS-aligned codes for ELA and Math.",
        "Forward Exam assessment specialist aligned to Wisconsin Academic Standards.",
        "Forward Exam Practice Questions"),
    "WY": _ccss_state("Wyoming", "Wyoming Content and Performance Standards", "WY-TOPP",
        "https://edu.wyoming.gov/",
        "You are supporting Wyoming K-12 educators. Align content to Wyoming Content and Performance Standards and WY-TOPP assessments. Use CCSS-aligned codes for ELA and Math.",
        "WY-TOPP assessment specialist aligned to Wyoming Content and Performance Standards.",
        "WY-TOPP Practice Questions"),
}


def _build_state_config():
    config = {}
    for code, s in STATE_DATA.items():
        config[code] = {
            "name": s["name"],
            "assessment_name": s["assessment_name"],
            "assessment_tab_label": s["assessment_name"] + " Prep",
            "assessment_hint": f"Generate standards-aligned practice questions for {s['assessment_name']} ({s['standards_name']}).",
            "standard_label": s["assessment_standard_label"],
            "standard_placeholder": s["assessment_example_placeholder"],
            "science_label": s["science_label"],
            "standards": {
                "Social Studies": {
                    "K-2": s.get("ss_standards", _GENERIC_SS_STDS)["K-2"],
                    "3-5": s.get("ss_standards", _GENERIC_SS_STDS)["3-5"],
                    "6-8": s.get("ss_standards", _GENERIC_SS_STDS)["6-8"],
                    "9-12": s.get("ss_standards", _GENERIC_SS_STDS)["9-12"],
                },
                "ELA": s["ela_standards"],
                "Math": s["math_standards"],
                "Science": s["science_standards"],
                "Health": {
                    "K-2": [
                        {"code":"HE.K.1","desc":"Personal Health: Identify behaviors that promote health and safety"},
                        {"code":"HE.1.2","desc":"Nutrition: Name food groups and their benefits for the body"},
                    ],
                    "3-5": [
                        {"code":"HE.3.1","desc":"Physical Activity: Explain how physical activity improves health"},
                        {"code":"HE.4.2","desc":"Safety: Describe strategies to prevent injuries and stay safe"},
                    ],
                    "6-8": [
                        {"code":"HE.6.1","desc":"Social/Emotional: Analyze how stress affects physical and mental health"},
                        {"code":"HE.7.3","desc":"Substance Prevention: Evaluate effects of alcohol, tobacco, and drugs on the body"},
                    ],
                    "9-12": [
                        {"code":"HE.9.1","desc":"Health Decisions: Evaluate how personal choices affect long-term health"},
                        {"code":"HE.10.2","desc":"Community Health: Analyze how public health policies affect individual wellness"},
                    ],
                },
                "World Languages": {
                    "K-2": [
                        {"code":"WL.K.1","desc":"Interpretive: Recognize familiar words and greetings in the target language"},
                        {"code":"WL.1.2","desc":"Cultural: Identify cultural practices and products from target language communities"},
                    ],
                    "3-5": [
                        {"code":"WL.3.1","desc":"Interpersonal: Exchange basic personal information using memorized phrases"},
                        {"code":"WL.4.2","desc":"Presentational: Present simple information about self and surroundings"},
                    ],
                    "6-8": [
                        {"code":"WL.6.1","desc":"Presentational: Present rehearsed information on familiar topics"},
                        {"code":"WL.7.2","desc":"Interpersonal: Participate in short conversations on familiar topics"},
                    ],
                    "9-12": [
                        {"code":"WL.9.1","desc":"Interpersonal: Sustain conversations on familiar and researched topics"},
                        {"code":"WL.10.2","desc":"Interpretive: Interpret main ideas and supporting details in authentic texts"},
                    ],
                },
                "Tech/CS": {
                    "K-2": [
                        {"code":"CS.K.1","desc":"Computing Systems: Select and use hardware and software for tasks"},
                        {"code":"CS.1.2","desc":"Algorithms: Create a sequence of steps to solve a simple problem"},
                    ],
                    "3-5": [
                        {"code":"CS.3.1","desc":"Algorithms: Create programs with sequences, events, and loops"},
                        {"code":"CS.4.2","desc":"Data: Organize and present data in visual formats"},
                    ],
                    "6-8": [
                        {"code":"CS.6.1","desc":"Programming: Create clearly named variables to store and manipulate data"},
                        {"code":"CS.7.2","desc":"Cybersecurity: Explain how encryption protects information"},
                    ],
                    "9-12": [
                        {"code":"CS.9.1","desc":"Data Analysis: Create data visualizations to communicate insights"},
                        {"code":"CS.10.2","desc":"Impact: Evaluate the social and ethical implications of computing"},
                    ],
                },
            },
        }
    return config


STATE_CONFIG = _build_state_config()

def _fetch_blog():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        sources = [
            "https://www.ed.gov/",
            "https://www.understood.org/",
        ]
        combined = ""
        for url in sources:
            try:
                r = requests.get(url, headers=headers, timeout=6)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                combined += text[:1500] + "\n---\n"
            except Exception:
                app.logger.warning("Failed to fetch blog content from %s", url, exc_info=True)
                continue
        if combined.strip():
            with _blog_lock:
                _cache["content"] = combined[:5000]
                _cache["last"] = time.time()
    except Exception:
        app.logger.exception("Blog content refresh failed")

def _bg_refresh():
    time.sleep(5)  # initial delay to avoid crash on import
    while True:
        _fetch_blog()
        time.sleep(3600)

threading.Thread(target=_bg_refresh, daemon=True).start()

def get_context(state="worldwide"):
    with _blog_lock:
        content = _cache["content"] if _cache["content"] else ""
    state_data = STATE_DATA.get(state, STATE_DATA["worldwide"])
    state_fallback = state_data["fallback"]
    if content:
        return content + "\n\nSTATE CONTEXT:\n" + state_fallback
    return state_fallback

# ── LLM ─────────────────────────────────────────────────────
def _focus_prompt(state="worldwide"):
    state_data = STATE_DATA.get(state, STATE_DATA["worldwide"])
    return state_data["focus_prompt"]


def _llm_via_groq(full_system, user):
    if not client:
        raise ValueError("GROQ_KEY not configured")
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": full_system}, {"role": "user", "content": user}],
        max_tokens=1500,
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").replace('**', '').strip()


def _llm_via_cerebras(full_system, user):
    key = os.environ.get('CEREBRAS_KEY')
    if not key:
        raise ValueError("CEREBRAS_KEY not configured")
    resp = requests.post(
        "https://api.cerebras.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_gemini(full_system, user):
    key = os.environ.get('GEMINI_KEY')
    if not key:
        raise ValueError("GEMINI_KEY not configured")
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
        headers={"Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": full_system + "\n\n" + user}]}],
            "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1500},
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return (text or "").replace('**', '').strip()


def _llm_via_cohere(full_system, user):
    key = os.environ.get('COHERE_KEY')
    if not key:
        raise ValueError("COHERE_KEY not configured")
    resp = requests.post(
        "https://api.cohere.com/v2/chat",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "command-r-plus",
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1500,
            "temperature": 0.6
        },
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data.get("message", {}).get("content", [{}])[0].get("text", "")
    return (text or "").replace('**', '').strip()
    

def _llm_via_mistral(full_system, user):
    key = os.environ.get('MISTRAL_KEY')
    if not key:
        raise ValueError("MISTRAL_KEY not configured")
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "mistral-small-latest",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_openrouter(full_system, user):
    key = os.environ.get('OPENROUTER_KEY')
    if not key:
        raise ValueError("OPENROUTER_KEY not configured")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://edusafeai.onrender.com",
            "X-Title": "EduSafeAI Hub",
        },
        json={
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=45,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


def _llm_via_huggingface(full_system, user):
    key = os.environ.get('HF_KEY')
    if not key:
        raise ValueError("HF_KEY not configured")
    resp = requests.post(
        "https://router.hugging-face.cn/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "messages": [{"role": "system", "content": full_system}, {"role": "user", "content": user}],
            "max_tokens": 1500,
            "temperature": 0.6,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").replace('**', '').strip()


# ── ORDERED FALLBACK DISPATCH ────────────────────────────────
_PROVIDERS = [
    ("groq", _llm_via_groq),
    ("cerebras", _llm_via_cerebras),
    ("gemini", _llm_via_gemini),
    ("cohere", _llm_via_cohere),
    ("mistral", _llm_via_mistral),
    ("openrouter", _llm_via_openrouter),
    ("huggingface", _llm_via_huggingface),
]

# ── RESPONSE CACHE ───────────────────────────────────────────
_resp_cache: collections.OrderedDict = collections.OrderedDict()
_CACHE_MAX = 500
_CACHE_TTL = 3600
_cache_lock = threading.Lock()

# ── RATE LIMITING ────────────────────────────────────────────
_RATE_LIMIT = 20       # requests per window per IP
_RATE_WINDOW = 60      # seconds
_rate_data: dict = {}
_rate_lock = threading.Lock()


def _check_rate_limit():
    ip = (request.access_route[0] if request.access_route else request.remote_addr) or '0.0.0.0'
    now = time.time()
    with _rate_lock:
        if ip not in _rate_data:
            _rate_data[ip] = collections.deque()
        dq = _rate_data[ip]
        while dq and dq[0] < now - _RATE_WINDOW:
            dq.popleft()
        if len(dq) >= _RATE_LIMIT:
            return False
        dq.append(now)
        # Clean up stale entries to prevent unbounded memory growth
        stale = [k for k, v in _rate_data.items() if not v]
        for k in stale:
            del _rate_data[k]
    return True


@app.before_request
def enforce_rate_limit():
    if request.method == 'POST':
        if not _check_rate_limit():
            return jsonify(error="Rate limit exceeded. Please wait a minute before making another request."), 429


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


def _cache_get(key):
    with _cache_lock:
        if key in _resp_cache:
            val, ts = _resp_cache[key]
            if time.time() - ts < _CACHE_TTL:
                _resp_cache.move_to_end(key)
                return val
            del _resp_cache[key]
    return None


def _cache_set(key, val):
    with _cache_lock:
        if key in _resp_cache:
            _resp_cache.move_to_end(key)
        _resp_cache[key] = (val, time.time())
        while len(_resp_cache) > _CACHE_MAX:
            _resp_cache.popitem(last=False)


def llm(system, user, state="worldwide"):
    focus_prompt = _focus_prompt(state)
    full_system = (
    system
    + "\n\n"
    + focus_prompt
    + "\n\nOUTPUT FORMAT RULES:\n"
    + "• Always respond with complete, well-structured output.\n"
    + "• Use clear section headers (e.g., using ALL CAPS or bold labels).\n"
    + "• Each section should contain 2-4 concise, actionable sentences or bullet points.\n"
    + "• Never truncate or cut off mid-sentence or mid-section.\n"
    + "• Do not repeat the user's question back to them.\n"
    + "• Do not add meta-commentary like \"Here is your lesson plan:\" — just output the content directly.\n\n"
    + "Reference context:\n"
    + get_context(state)
)

    cache_key = hashlib.sha256((system + user + state).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        app.logger.info("LLM request served from cache")
        return cached

    for name, func in _PROVIDERS:
        try:
            result = func(full_system, user)
            if result:
                _cache_set(cache_key, result)
                app.logger.info("LLM request served by %s", name)
                return result
        except Exception as exc:
            app.logger.warning("Provider %s failed: %s", name, exc)

    return "⚠️ All AI providers are currently busy. Please try again in a few minutes."


_MAX_FIELD_LEN = 4000  # max characters per input field


def _get_json(required_fields):
    data = request.get_json(silent=True) or {}
    missing = [field for field in required_fields if not str(data.get(field, "")).strip()]
    if missing:
        return None, jsonify(error=f"Missing required field(s): {', '.join(missing)}"), 400
    for field, val in data.items():
        if isinstance(val, str) and len(val) > _MAX_FIELD_LEN:
            return None, jsonify(error=f"Field '{field}' exceeds maximum length of {_MAX_FIELD_LEN} characters."), 400
    return data, None, None


def _internal_error():
    app.logger.exception("Unhandled route error")
    return jsonify(error="Internal server error. Please try again."), 500


# ── HTML ─────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<title>EduSafeAI Hub</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="AI-powered tools for K-12 educators across the United States.">
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%8F%AB%3C/text%3E%3C/svg%3E">
<style>
:root{--gd:#1a472a;--gm:#2d6a4f;--gl:#52b788;--gp:#f0f7f0;--gb:#c8e6c9;--w:#fff;--gray:#666;--r:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:var(--gp);color:#222}
.header{background:linear-gradient(135deg,var(--gd),var(--gl));color:#fff;padding:32px 20px;text-align:center}
.header h1{font-size:2.2em;margin-bottom:8px;letter-spacing:1px}
.header p{font-size:1em;opacity:.92;max-width:580px;margin:0 auto}
.badges{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:14px}
.badge{background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:5px 14px;font-size:.82em}
.container{max-width:920px;margin:28px auto;padding:0 16px}
.tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(80px,1fr));gap:8px;margin-bottom:24px}
.tabs button{background:var(--gm);color:#fff;border:none;padding:10px 4px;border-radius:var(--r);cursor:pointer;font-size:11px;font-weight:600;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:3px;width:100%}
.tabs button:hover{background:var(--gd);transform:translateY(-2px)}
.tabs button.active{background:var(--gd);border-bottom:3px solid var(--gl)}
.tab-icon{font-size:1.4em}
.tab{display:none}.tab.active{display:block;animation:fadeIn .35s}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.card{background:var(--w);padding:28px;border-radius:16px;box-shadow:0 4px 20px rgba(0,0,0,.09);margin-bottom:4px}
.card h2{color:var(--gd);margin-bottom:8px;font-size:1.35em;display:flex;align-items:center;gap:8px}
.hint{color:var(--gray);font-size:.87em;margin-bottom:18px;background:#e8f5e9;padding:12px 14px;border-radius:0 10px 10px 0;border-left:4px solid var(--gl)}
.new-badge{background:#e8f5e9;color:#2d6a4f;border:1px solid #a5d6a7;border-radius:12px;padding:2px 8px;font-size:.7em;font-weight:bold;margin-left:8px;vertical-align:middle}
.form-row{display:grid;grid-template-columns:1fr;gap:14px;margin-bottom:8px}
@media(min-width:500px){.form-row.two{grid-template-columns:1fr 1fr}}
.field{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
label{font-weight:600;color:var(--gd);font-size:.9em;display:flex;align-items:center;gap:6px}
.tip{display:inline-block;background:var(--gm);color:#fff;border-radius:50%;width:17px;height:17px;font-size:.7em;text-align:center;line-height:17px;cursor:help;position:relative}
.tip:hover::after{content:attr(data-tip);position:absolute;left:22px;top:-4px;background:#333;color:#fff;padding:6px 10px;border-radius:6px;font-size:12px;white-space:nowrap;z-index:10;font-weight:normal}
input,select,textarea{width:100%;padding:11px 13px;border:1.5px solid #ddd;border-radius:var(--r);font-size:14px;transition:border .2s;background:#fafafa}
input:focus,select:focus,textarea:focus{border-color:var(--gl);outline:none;background:#fff}
textarea{resize:vertical;min-height:100px}
.std-desc{background:#e8f5e9;border-left:3px solid var(--gl);padding:8px 12px;border-radius:0 8px 8px 0;font-size:13px;color:#2d4a35;margin-top:6px;display:none}
.std-desc.show{display:block}
.btn{background:linear-gradient(135deg,#43a047,#2d6a4f);color:#fff;border:none;padding:14px;width:100%;border-radius:var(--r);font-size:15px;cursor:pointer;margin:14px 0 8px;font-weight:bold;letter-spacing:.5px;transition:all .2s;box-shadow:0 3px 8px rgba(0,0,0,.15)}
.btn:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.2)}
.btn:disabled{opacity:.6;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:6px}
.output{background:#f6fdf6;border:1.5px solid var(--gb);border-radius:var(--r);padding:18px;min-height:90px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:8px;right:8px;background:var(--gm);color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
hr{border:none;border-top:1px solid #e0e0e0;margin:18px 0}
.footer{text-align:center;padding:28px 16px;color:var(--gray);font-size:13px;line-height:2;background:var(--w);border-radius:16px;margin-top:20px}
select optgroup{font-weight:bold;color:var(--gd)}
.char-counter{font-size:.78em;color:var(--gray);text-align:right;margin-top:2px}
.state-selector-bar{background:linear-gradient(135deg,#1b4332,#2d6a4f);padding:12px 20px;display:flex;align-items:center;justify-content:center;gap:16px;flex-wrap:wrap;border-bottom:3px solid #52b788}
.state-selector-bar label{color:#fff;font-weight:700;font-size:.95em;white-space:nowrap}
#stateSelect{width:auto;min-width:220px;max-width:280px;padding:9px 14px;border:2px solid #52b788;border-radius:20px;font-size:.95em;font-weight:600;background:#fff;color:#1a472a;cursor:pointer}
.state-banner{background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.4);border-radius:20px;padding:5px 16px;font-size:.85em;font-weight:600}
</style>
</head>
<body>
<div class="header">
  <a href="/" style="color:inherit;text-decoration:none"><h1>🛡️ EduSafeAI Hub</h1></a>
  <p><b>AI tools for K-12 educators worldwide</b></p>
  <div class="badges">
    <span class="badge">🏫 Education-Ready</span>
    <span class="badge">♿ IEP & ELL Ready</span>
    <span class="badge">🔒 No Student Data Stored</span>
    <span class="badge">☁️ Multi-AI Powered</span>
  </div>
</div>

<div class="state-selector-bar">
  <label for="stateSelect">&#x1F5FA;&#xFE0F; Select Your Region:</label>
  <select id="stateSelect" onchange="onStateChange()">
    <option value="worldwide">&#x1F30D; Worldwide (CCSS/NGSS)</option>
    <option value="AL">Alabama</option>
    <option value="AK">Alaska</option>
    <option value="AZ">Arizona</option>
    <option value="AR">Arkansas</option>
    <option value="CA">California</option>
    <option value="CO">Colorado</option>
    <option value="CT">Connecticut</option>
    <option value="DE">Delaware</option>
    <option value="FL">Florida</option>
    <option value="GA">Georgia</option>
    <option value="HI">Hawaii</option>
    <option value="ID">Idaho</option>
    <option value="IL">Illinois</option>
    <option value="IN">Indiana</option>
    <option value="IA">Iowa</option>
    <option value="KS">Kansas</option>
    <option value="KY">Kentucky</option>
    <option value="LA">Louisiana</option>
    <option value="ME">Maine</option>
    <option value="MD">Maryland</option>
    <option value="MA">Massachusetts</option>
    <option value="MI">Michigan</option>
    <option value="MN">Minnesota</option>
    <option value="MS">Mississippi</option>
    <option value="MO">Missouri</option>
    <option value="MT">Montana</option>
    <option value="NE">Nebraska</option>
    <option value="NV">Nevada</option>
    <option value="NH">New Hampshire</option>
    <option value="NJ">New Jersey</option>
    <option value="NM">New Mexico</option>
    <option value="NY">New York</option>
    <option value="NC">North Carolina</option>
    <option value="ND">North Dakota</option>
    <option value="OH">Ohio</option>
    <option value="OK">Oklahoma</option>
    <option value="OR">Oregon</option>
    <option value="PA">Pennsylvania</option>
    <option value="RI">Rhode Island</option>
    <option value="SC">South Carolina</option>
    <option value="SD">South Dakota</option>
    <option value="TN">Tennessee</option>
    <option value="TX">Texas</option>
    <option value="UT">Utah</option>
    <option value="VT">Vermont</option>
    <option value="VA">Virginia</option>
    <option value="WA">Washington</option>
    <option value="WV">West Virginia</option>
    <option value="WI">Wisconsin</option>
    <option value="WY">Wyoming</option>
  </select>
  <span id="state-banner" class="state-banner">&#x1F30D; Worldwide (CCSS/NGSS) Standards</span>
</div>

<div class="container">
  <div class="tabs" id="tool-tabs" role="tablist" aria-label="Tool categories">
    <button class="active" role="tab" aria-selected="true" aria-controls="lesson" data-tab="lesson"><span class="tab-icon">📖</span>Lesson</button>
    <button role="tab" aria-selected="false" aria-controls="feedback" data-tab="feedback"><span class="tab-icon">💬</span>Feedback</button>
    <button role="tab" aria-selected="false" aria-controls="diff" data-tab="diff"><span class="tab-icon">♿</span>IEP/ELL</button>
    <button role="tab" aria-selected="false" aria-controls="policy" data-tab="policy"><span class="tab-icon">📄</span>Policy</button>
    <button role="tab" aria-selected="false" aria-controls="email" data-tab="email"><span class="tab-icon">✉️</span>Email</button>
    <button role="tab" aria-selected="false" aria-controls="integrity" data-tab="integrity"><span class="tab-icon">🧪</span>AI Integrity</button>
    <button role="tab" aria-selected="false" aria-controls="assessment_prep" data-tab="assessment_prep"><span class="tab-icon">📊</span>Assessment Prep</button>
    <button role="tab" aria-selected="false" aria-controls="parent" data-tab="parent"><span class="tab-icon">🗣️</span>Parent Letter</button>
    <button role="tab" aria-selected="false" aria-controls="unit" data-tab="unit"><span class="tab-icon">📅</span>Unit Plan</button>
    <button role="tab" aria-selected="false" aria-controls="iep504" data-tab="iep504"><span class="tab-icon">🏫</span>504/IEP</button>
    <button role="tab" aria-selected="false" aria-controls="quiz" data-tab="quiz"><span class="tab-icon">❓</span>Quiz</button>
    <button role="tab" aria-selected="false" aria-controls="rubric" data-tab="rubric"><span class="tab-icon">🧾</span>Rubric</button>
    <button role="tab" aria-selected="false" aria-controls="refine" data-tab="refine"><span class="tab-icon">🔧</span>Improve</button>
    <button role="tab" aria-selected="false" aria-controls="sitefb" data-tab="sitefb"><span class="tab-icon">⭐</span>Contact Us</button>
  </div>

  <!-- 1. LESSON -->
  <div id="lesson" class="tab active"><div class="card">
    <h2>📖 AI Lesson Designer</h2>
    <p class="hint">Generate a complete standards-aligned lesson plan with hooks, activities, and exit tickets.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="l2" onchange="updateStandards()">
          <option value="K-2">K-2</option><option value="3-5">3-5</option>
          <option value="6-8" selected>6-8</option><option value="9-12">9-12</option>
        </select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="l3" onchange="updateStandards()">
          <option value="Social Studies" selected>Social Studies</option>
          <option value="ELA">ELA</option><option id="science-option" value="Science">Science</option>
          <option value="Math">Math</option><option value="Health">Health</option>
          <option value="World Languages">World Languages</option><option value="Tech/CS">Tech/CS</option>
        </select>
      </div>
    </div>
    <div class="field">
      <label>Standard <span class="tip" data-tip="Optional — AI will auto-select if blank">?</span></label>
      <select id="l1" onchange="showStdDesc()">
        <option value="">-- Select grade & subject above first --</option>
      </select>
      <div id="std-desc" class="std-desc"></div>
    </div>
    <div class="field">
      <label>Duration</label>
      <select id="l4"><option selected>45 min</option><option>60 min</option><option>90 min</option><option>Block</option></select>
    </div>
    <button class="btn" id="lb" onclick="call('/lesson',{standard:g('l1'),grade:g('l2'),subject:g('l3'),duration:g('l4')},'lo','lb','🎯 Generate Full Lesson Plan')">🎯 Generate Full Lesson Plan</button>
    <div class="output-wrap"><div id="lo" class="output">Your lesson plan will appear here...</div><button class="copy-btn" onclick="cp('lo')">📋 Copy</button></div>
  </div></div>

  <!-- 2. FEEDBACK -->
  <div id="feedback" class="tab"><div class="card">
    <h2>💬 Student Feedback Generator</h2>
    <p class="hint">Get constructive, rubric-aligned feedback without rewriting student work.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="f1"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option></select>
      </div>
      <div class="field">
        <label>Rubric Type</label>
        <select id="f2"><option selected>General Writing</option><option>Social Studies DBQ</option><option>ELA Essay</option><option>Science Lab Report</option></select>
      </div>
    </div>
    <div class="field">
      <label>Student Work <span class="tip" data-tip="Paste student writing below">?</span></label>
      <textarea id="f3" rows="6" maxlength="4000" placeholder="Paste student work here..."></textarea>
      <span class="char-counter"></span>
    </div>
    <button class="btn" id="fb" onclick="call('/feedback',{work:g('f3'),grade:g('f1'),rubric:g('f2')},'fo','fb','💬 Generate Feedback')">💬 Generate Feedback</button>
    <div class="output-wrap"><div id="fo" class="output">Feedback will appear here...</div><button class="copy-btn" onclick="cp('fo')">📋 Copy</button></div>
  </div></div>

  <!-- 3. IEP/ELL -->
  <div id="diff" class="tab"><div class="card">
    <h2>♿ IEP/ELL Differentiator</h2>
    <p class="hint">Adapt any lesson for students with IEPs, ELL needs, ADHD, dyslexia, and more.</p>
    <hr>
    <div class="field">
      <label>Paste Your Lesson</label>
      <textarea id="d1" rows="5" maxlength="4000" placeholder="Paste any lesson plan here..."></textarea>
      <span class="char-counter"></span>
    </div>
    <div class="field">
      <label>Student Needs</label>
      <input id="d2" placeholder="e.g. ELL Level 2, ADHD, dyslexia, hearing impaired">
    </div>
    <button class="btn" id="db" onclick="call('/differentiate',{lesson:g('d1'),needs:g('d2')},'do','db','♿ Differentiate This Lesson')">♿ Differentiate This Lesson</button>
    <div class="output-wrap"><div id="do" class="output">Differentiated version will appear here...</div><button class="copy-btn" onclick="cp('do')">📋 Copy</button></div>
  </div></div>

  <!-- 4. POLICY -->
  <div id="policy" class="tab"><div class="card">
    <h2>📄 AI Policy Generator</h2>
    <p class="hint">Generate an official AI use policy for your school or district.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>School / District</label><input id="p1" placeholder="Enter your school name"></div>
      <div class="field"><label>School Year</label><input id="p2" value="2025-2026"></div>
    </div>
    <div class="form-row two">
      <div class="field">
        <label>Grade Band</label>
        <select id="p3"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option><option>All grades</option></select>
      </div>
      <div class="field"><label>Main Concerns</label><input id="p4" placeholder="e.g. plagiarism, data privacy, parent communication"></div>
    </div>
    <button class="btn" id="pb" onclick="call('/policy',{school:g('p1'),year:g('p2'),grade:g('p3'),concerns:g('p4')},'po','pb','📄 Generate Official Policy')">📄 Generate Official Policy</button>
    <div class="output-wrap"><div id="po" class="output">Policy will appear here...</div><button class="copy-btn" onclick="cp('po')">📋 Copy</button></div>
  </div></div>

  <!-- 5. EMAIL -->
  <div id="email" class="tab"><div class="card">
    <h2>✉️ Email Drafter</h2>
    <p class="hint">Draft professional emails to parents, admin, or colleagues in seconds.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Recipient</label>
        <select id="e1"><option selected>Parent</option><option>Principal</option><option>Colleague</option><option>Superintendent</option><option>IEP Team</option><option>School Board</option></select>
      </div>
      <div class="field">
        <label>Tone</label>
        <select id="e2"><option selected>professional</option><option>friendly</option><option>formal</option><option>urgent</option><option>empathetic</option></select>
      </div>
    </div>
    <div class="field"><label>What is the email about?</label><input id="e3" placeholder="e.g. Student used AI without permission on assignment"></div>
    <button class="btn" id="eb" onclick="call('/email',{recipient:g('e1'),tone:g('e2'),topic:g('e3')},'eo','eb','✉️ Draft Email')">✉️ Draft Email</button>
    <div class="output-wrap"><div id="eo" class="output">Email will appear here...</div><button class="copy-btn" onclick="cp('eo')">📋 Copy</button></div>
  </div></div>

  <!-- 6. AI INTEGRITY -->
  <div id="integrity" class="tab"><div class="card">
    <h2>🧪 AI Integrity Checker <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Paste student work to get an AI-risk analysis plus a conversation script to discuss it professionally.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="i1"><option>K-2</option><option>3-5</option><option>6-8</option><option selected>9-12</option></select>
      </div>
      <div class="field">
        <label>Assignment Type</label>
        <select id="i2"><option selected>Essay</option><option>Research Paper</option><option>Short Answer</option><option>Lab Report</option><option>Creative Writing</option></select>
      </div>
    </div>
    <div class="field"><label>Student Work</label><textarea id="i3" rows="7" maxlength="4000" placeholder="Paste student work here..."></textarea><span class="char-counter"></span></div>
    <button class="btn" id="ib" onclick="call('/integrity',{work:g('i3'),grade:g('i1'),type:g('i2')},'io','ib','🧪 Check AI Integrity')">🧪 Check AI Integrity</button>
    <div class="output-wrap"><div id="io" class="output">Analysis will appear here...</div><button class="copy-btn" onclick="cp('io')">📋 Copy</button></div>
  </div></div>

  <!-- 7. ASSESSMENT PREP -->
  <div id="assessment_prep" class="tab"><div class="card">
    <h2 id="assess-title">📊 Standards-Based Assessment Prep <span class="new-badge">UNIQUE</span></h2>
    <p id="assess-hint" class="hint">Generate standards-aligned practice questions for any assessment framework.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="n1"><option>3</option><option>4</option><option>5</option><option>6</option><option>7</option><option selected>8</option><option>9</option><option>10</option><option>11</option></select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="n2"><option selected>ELA</option><option>Math</option><option id="assess-science-option">Science (NGSS)</option></select>
      </div>
    </div>
    <div class="form-row two">
      <div class="field"><label id="assess-std-label">Standard <span class="tip" data-tip="Leave blank for auto-select">?</span></label><input id="n3" placeholder="e.g. RL.8.1 or leave blank"></div>
      <div class="field">
        <label>Question Type</label>
        <select id="n4"><option selected>Multiple Choice</option><option>Short Answer</option><option>Evidence-Based</option><option>Mixed</option></select>
      </div>
    </div>
    <div class="field">
      <label>Number of Questions</label>
      <select id="n5"><option selected>5</option><option>10</option><option>15</option></select>
    </div>
    <button class="btn" id="nb" onclick="call('/assessment_prep',{grade:g('n1'),subject:g('n2'),standard:g('n3'),qtype:g('n4'),num:g('n5')},'no','nb','📊 Generate Assessment Practice')">📊 Generate Assessment Practice Questions</button>
    <div class="output-wrap"><div id="no" class="output">Practice questions will appear here...</div><button class="copy-btn" onclick="cp('no')">📋 Copy</button></div>
  </div></div>

  <!-- 8. PARENT LETTER -->
  <div id="parent" class="tab"><div class="card">
    <h2>🗣️ Parent Letter Generator <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Generate culturally appropriate parent letters in multiple languages.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Language</label>
        <select id="pl1"><option selected>English</option><option>Spanish</option><option>Portuguese</option><option>French</option><option>Chinese (Simplified)</option><option>Arabic</option><option>Haitian Creole</option></select>
      </div>
      <div class="field">
        <label>Letter Type</label>
        <select id="pl2"><option selected>Classroom AI Policy</option><option>Student Behavior</option><option>Academic Progress</option><option>IEP Meeting Invite</option><option>Field Trip Permission</option><option>Homework Policy</option></select>
      </div>
    </div>
    <div class="field"><label>Key Details</label><input id="pl3" placeholder="e.g. Meeting on March 5th at 3pm"></div>
    <div class="field"><label>Teacher Name & School</label><input id="pl4" placeholder="e.g. Ms. Johnson, Lincoln Elementary"></div>
    <button class="btn" id="plb" onclick="call('/parent_letter',{lang:g('pl1'),type:g('pl2'),details:g('pl3'),teacher:g('pl4')},'plo','plb','🗣️ Generate Parent Letter')">🗣️ Generate Parent Letter</button>
    <div class="output-wrap"><div id="plo" class="output">Parent letter will appear here...</div><button class="copy-btn" onclick="cp('plo')">📋 Copy</button></div>
  </div></div>

  <!-- 9. UNIT PLANNER -->
  <div id="unit" class="tab"><div class="card">
    <h2>📅 2-Week Unit Planner <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Plan a complete 2-week unit with daily breakdown, assessments, and differentiation.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Unit Topic</label><input id="u1" placeholder="e.g. American Revolution, Fractions, Ecosystems"></div>
      <div class="field"><label>Standard <span class="tip" data-tip="Optional — AI will auto-select if blank">?</span></label><input id="u2" placeholder="e.g. 6.1.8.HistoryCC.3 or local standard"></div>
    </div>
    <div class="form-row two">
      <div class="field">
        <label>Grade</label>
        <select id="u3"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select>
      </div>
      <div class="field">
        <label>Subject</label>
        <select id="u4"><option selected>Social Studies</option><option>ELA</option><option>Science</option><option>Math</option><option>Health</option></select>
      </div>
    </div>
    <div class="field">
      <label>Class Duration</label>
      <select id="u5"><option selected>45 min</option><option>60 min</option><option>90 min</option><option>Block</option></select>
    </div>
    <button class="btn" id="ub" onclick="call('/unit_plan',{topic:g('u1'),standard:g('u2'),grade:g('u3'),subject:g('u4'),duration:g('u5')},'uo','ub','📅 Generate 2-Week Unit Plan')">📅 Generate 2-Week Unit Plan</button>
    <div class="output-wrap"><div id="uo" class="output">Unit plan will appear here...</div><button class="copy-btn" onclick="cp('uo')">📋 Copy</button></div>
  </div></div>

  <!-- 10. 504/IEP -->
  <div id="iep504" class="tab"><div class="card">
    <h2>🏫 504 vs IEP Helper <span class="new-badge">UNIQUE</span></h2>
    <p class="hint">Clarify 504 vs IEP differences and generate accommodations for any disability or need.</p>
    <hr>
    <div class="form-row two">
      <div class="field">
        <label>Plan Type</label>
        <select id="s1">
          <option selected>Explain difference: 504 vs IEP</option>
          <option>Generate 504 accommodations</option>
          <option>Generate IEP accommodations</option>
          <option>Generate both 504 & IEP options</option>
        </select>
      </div>
      <div class="field">
        <label>Grade</label>
        <select id="s2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select>
      </div>
    </div>
    <div class="field"><label>Student Disability / Need</label><input id="s3" placeholder="e.g. ADHD, anxiety, dyslexia, autism, hearing impaired"></div>
    <div class="field"><label>Subject Context</label><input id="s4" placeholder="e.g. ELA class, standardized testing, all subjects"></div>
    <button class="btn" id="sb" onclick="call('/iep504',{plan:g('s1'),grade:g('s2'),disability:g('s3'),context:g('s4')},'so','sb','🏫 Generate Accommodations')">🏫 Generate Accommodations</button>
    <div class="output-wrap"><div id="so" class="output">Accommodations will appear here...</div><button class="copy-btn" onclick="cp('so')">📋 Copy</button></div>
  </div></div>


  <!-- 11. QUIZ BUILDER -->
  <div id="quiz" class="tab"><div class="card">
    <h2>❓ Quiz Builder</h2>
    <p class="hint">Create standards-aligned formative quizzes with answer keys.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Topic</label><input id="q1" placeholder="e.g. Civil War causes"></div>
      <div class="field"><label>Grade</label><select id="q2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select></div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Question Type</label><select id="q3"><option selected>Mixed</option><option>Multiple Choice</option><option>Short Answer</option></select></div>
      <div class="field"><label>Number of Questions</label><select id="q4"><option>5</option><option selected>10</option><option>15</option></select></div>
    </div>
    <button class="btn" id="qb" onclick="call('/quiz',{topic:g('q1'),grade:g('q2'),qtype:g('q3'),num:g('q4')},'qo','qb','❓ Generate Quiz')">❓ Generate Quiz</button>
    <div class="output-wrap"><div id="qo" class="output">Quiz will appear here...</div><button class="copy-btn" onclick="cp('qo')">📋 Copy</button></div>
  </div></div>

  <!-- 12. RUBRIC BUILDER -->
  <div id="rubric" class="tab"><div class="card">
    <h2>🧾 Rubric Builder</h2>
    <p class="hint">Generate clear performance-level rubrics aligned to your assignment goals.</p>
    <hr>
    <div class="form-row two">
      <div class="field"><label>Assignment</label><input id="r1" placeholder="e.g. Argument essay"></div>
      <div class="field"><label>Grade</label><select id="r2"><option>K-2</option><option>3-5</option><option selected>6-8</option><option>9-12</option></select></div>
    </div>
    <div class="form-row two">
      <div class="field"><label>Criteria Count</label><select id="r3"><option>3</option><option selected>4</option><option>5</option></select></div>
      <div class="field"><label>Scale</label><select id="r4"><option selected>4-point</option><option>5-point</option></select></div>
    </div>
    <button class="btn" id="rb" onclick="call('/rubric',{assignment:g('r1'),grade:g('r2'),criteria:g('r3'),scale:g('r4')},'ro','rb','🧾 Generate Rubric')">🧾 Generate Rubric</button>
    <div class="output-wrap"><div id="ro" class="output">Rubric will appear here...</div><button class="copy-btn" onclick="cp('ro')">📋 Copy</button></div>
  </div></div>

  <!-- 13. IMPROVE AI RESPONSE -->
  <div id="refine" class="tab"><div class="card">
    <h2>🔧 Improve AI Response</h2>
    <p class="hint">Ask the AI to revise any answer — make it shorter, simpler, add accommodations, etc.</p>
    <hr>
    <div class="field"><label>AI Output to Improve</label><textarea id="rf1" rows="6" maxlength="4000" placeholder="Paste the AI response you want to improve..."></textarea><span class="char-counter"></span></div>
    <div class="field"><label>What should be fixed?</label><input id="rf2" placeholder="e.g. make it shorter, add accommodations, simpler language"></div>
    <button class="btn" id="rfb" onclick="call('/refine_response',{response:g('rf1'),request:g('rf2')},'rfo','rfb','🔧 Improve Response')">🔧 Improve Response</button>
    <div class="output-wrap"><div id="rfo" class="output">Refined response will appear here...</div><button class="copy-btn" onclick="cp('rfo')">📋 Copy</button></div>
  </div></div>

  <!-- 14. CONTACT US -->
  <div id="sitefb" class="tab"><div class="card">
    <h2>⭐ Contact Us</h2>
    <p>Have ideas, suggestions, or found a bug? We'd love to hear from you!</p>
    <hr>
    <p style="font-size:1.1em">📧 Email us at: <strong>admin@edusafeai.com</strong></p>
  </div></div>

</div>

<div class="footer">
  <strong>🛡️ EduSafeAI Hub</strong> | AI tools for K-12 educators worldwide <br>
  🔒 EduSafeAI does not store student data. Please do not enter personally identifiable student information.<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ AI-generated content may contain inaccuracies. 
    This tool is for informational purposes only and does not constitute legal, medical, or professional advice. 
    Always consult qualified professionals for official IEP, 504, or compliance decisions.
  </span>
</div>

<script>
const STATE_CONFIG = {{ state_config_json | safe }};

let currentState = 'worldwide';
let STANDARDS = STATE_CONFIG['worldwide'].standards;

function g(id){return document.getElementById(id).value;}

function updateStandards(){
  const grade=g('l2'),subject=g('l3'),sel=document.getElementById('l1'),desc=document.getElementById('std-desc');
  sel.innerHTML='';desc.className='std-desc';desc.textContent='';
  const stds=STATE_CONFIG[currentState]?STATE_CONFIG[currentState].standards:STANDARDS;
  const list=(stds[subject]||{})[grade]||[];
  if(!list.length){sel.innerHTML='<option value="">-- No standards available --</option>';return;}
  list.forEach((s,i)=>{const o=document.createElement('option');o.value=s.code;o.textContent=s.code+' \u2014 '+s.desc;if(i===0)o.selected=true;sel.appendChild(o);});
  showStdDesc();
}

function showStdDesc(){
  const grade=g('l2'),subject=g('l3'),code=g('l1'),desc=document.getElementById('std-desc');
  const stds=STATE_CONFIG[currentState]?STATE_CONFIG[currentState].standards:STANDARDS;
  const list=(stds[subject]||{})[grade]||[];
  const found=list.find(s=>s.code===code);
  if(found){desc.textContent=found.desc;desc.className='std-desc show';}
  else{desc.className='std-desc';}
}

function onStateChange(){
  const sel=document.getElementById('stateSelect');
  currentState=sel.value||'worldwide';
  const cfg=STATE_CONFIG[currentState]||STATE_CONFIG['worldwide'];
  STANDARDS=cfg.standards;
  document.getElementById('state-banner').textContent=cfg.name+' Standards';
  const assessBtn=document.querySelector('[data-tab="assessment_prep"]');
  if(assessBtn)assessBtn.innerHTML='<span class="tab-icon">&#x1F4CA;</span>'+cfg.assessment_name;
  const assessTitle=document.getElementById('assess-title');
  if(assessTitle)assessTitle.innerHTML='&#x1F4CA; '+cfg.assessment_name+' Practice Prep <span class="new-badge">UNIQUE</span>';
  const assessHint=document.getElementById('assess-hint');
  if(assessHint)assessHint.textContent=cfg.assessment_hint;
  const assessStdLabel=document.getElementById('assess-std-label');
  if(assessStdLabel)assessStdLabel.innerHTML=cfg.standard_label+' <span class="tip" data-tip="Leave blank for auto-select">?</span>';
  const n3=document.getElementById('n3');
  if(n3)n3.placeholder=cfg.standard_placeholder;
  const assessSciOption=document.getElementById('assess-science-option');
  if(assessSciOption)assessSciOption.textContent=cfg.science_label;
  const u2=document.getElementById('u2');
  if(u2)u2.placeholder=cfg.standard_placeholder;
  updateStandards();
}

window.onload=()=>{updateStandards();onStateChange();};

function show(tab,btn){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button[role="tab"]').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');});
  const target=document.getElementById(tab);
  if(target){target.classList.add('active');}
  if(btn){btn.classList.add('active');btn.setAttribute('aria-selected','true');}
}

function _fallbackCopy(text){
  const ta=document.createElement('textarea');
  ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.focus();ta.select();
  try{document.execCommand('copy');}catch(e){}
  document.body.removeChild(ta);
}

function cp(id){
  const text=document.getElementById(id).innerText;
  const btn=document.querySelector('#'+id).parentNode.querySelector('.copy-btn');
  const done=()=>{btn.textContent='✅ Copied!';setTimeout(()=>btn.textContent='📋 Copy',2000);};
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(()=>{_fallbackCopy(text);done();});
  }else{_fallbackCopy(text);done();}
}

let focusMode='general';

document.addEventListener('DOMContentLoaded',()=>{
  // Tab delegation
  const tabList=document.getElementById('tool-tabs');
  tabList.addEventListener('click',e=>{
    const btn=e.target.closest('button[data-tab]');
    if(btn)show(btn.dataset.tab,btn);
  });
  // Keyboard navigation for tabs (ARIA tablist pattern)
  tabList.addEventListener('keydown',e=>{
    const tabs=Array.from(tabList.querySelectorAll('button[role="tab"]'));
    const idx=tabs.indexOf(document.activeElement);
    if(idx===-1)return;
    let next=-1;
    if(e.key==='ArrowRight'||e.key==='ArrowDown')next=(idx+1)%tabs.length;
    else if(e.key==='ArrowLeft'||e.key==='ArrowUp')next=(idx-1+tabs.length)%tabs.length;
    else if(e.key==='Home')next=0;
    else if(e.key==='End')next=tabs.length-1;
    if(next!==-1){e.preventDefault();tabs[next].focus();show(tabs[next].dataset.tab,tabs[next]);}
  });
  // Character counters
  document.querySelectorAll('textarea[maxlength]').forEach(ta=>{
    const counter=ta.nextElementSibling;
    if(counter&&counter.classList.contains('char-counter')){
      const max=ta.getAttribute('maxlength');
      counter.textContent=`0 / ${max}`;
      ta.addEventListener('input',()=>{counter.textContent=`${ta.value.length} / ${max}`;});
    }
  });
});

async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId),btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Generating...';
  out.textContent='⏳ AI is thinking...';
  try{
    const state=currentState||'worldwide';
    const payload={...data,focus:focusMode,state:state};
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(!r.ok){
      let msg='Request failed.';
      try{const ej=await r.json();msg=ej.error||ej.result||msg;}catch(_){const t=await r.text();msg=t.substring(0,300)||msg;}
      out.textContent='❌ '+msg;
      return;
    }
    const j=await r.json();
    out.textContent=j.result || j.error || 'No response content.';
  }catch(e){
    out.textContent='❌ Error: '+e.message;
  }finally{
    btn.disabled=false;btn.textContent=label;
  }
}
</script>
</body>
</html>"""

# ── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML, state_config_json=json.dumps(STATE_CONFIG))


@app.route('/health')
def health():
    return jsonify(status='ok')

@app.route('/lesson', methods=['POST'])
def do_lesson():
    try:
        d, err, code = _get_json(["grade", "subject", "duration"])
        if err:
            return err, code
        standard = d.get('standard', '').strip()
        std = standard if standard else "appropriate grade-level standard"
        return jsonify(result=llm(
            f"Expert {d['grade']} {d['subject']} teacher.",
            f"Create a {d['duration']} lesson for standard/topic: {std}.\n\n🎣 HOOK\n📚 MAIN ACTIVITY (ELL/IEP notes)\n🔍 ORIGINALITY CHECK\n🎯 EXIT TICKET",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/feedback', methods=['POST'])
def do_feedback():
    try:
        d, err, code = _get_json(["work", "grade", "rubric"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Encouraging {d['grade']} teacher. NEVER rewrite student work.",
            f"Rubric: {d['rubric']}\n\nStudent work:\n{d['work']}\n\n💪 STRENGTH\n📖 EVIDENCE\n🧠 REASONING\n🎯 NEXT STEP\n⚠️ ACADEMIC INTEGRITY NOTE",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/differentiate', methods=['POST'])
def do_diff():
    try:
        d, err, code = _get_json(["lesson", "needs"])
        if err:
            return err, code
        return jsonify(result=llm(
            "SPED and ELL expert teacher.",
            f"Adapt this lesson for: {d['needs']}\n\n{d['lesson']}\n\n📝 SIMPLIFIED VERSION\n🖼️ VISUAL AIDS\n🪜 SCAFFOLDS\n📊 MODIFIED ASSESSMENT\n⏱️ EXTENDED TIME NOTES",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/policy', methods=['POST'])
def do_policy():
    try:
        d, err, code = _get_json(["school", "year", "grade", "concerns"])
        if err:
            return err, code
        return jsonify(result=llm(
            "School administrator and education policy expert.",
            f"Write official AI use policy for {d['school']} | Year: {d['year']} | Grade: {d['grade']} | Concerns: {d['concerns']}\n\n📌 PURPOSE\n📋 SCOPE\n👨‍🎓 STUDENT GUIDELINES\n👩‍🏫 TEACHER RESPONSIBILITIES\n👨‍👩‍👧 PARENT COMMUNICATION\n⚠️ CONSEQUENCES\n📅 REVIEW DATE",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/email', methods=['POST'])
def do_email():
    try:
        d, err, code = _get_json(["recipient", "tone", "topic"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Professional educator communication expert.",
            f"Write a {d['tone']} email to {d['recipient']} about: {d['topic']}.\nInclude: Subject line, greeting, 2-3 paragraphs, professional closing. Under 200 words.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/integrity', methods=['POST'])
def do_integrity():
    try:
        d, err, code = _get_json(["work", "grade", "type"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Educator and academic integrity specialist.",
            f"Grade: {d['grade']} | Assignment: {d['type']}\n\nStudent work:\n{d['work']}\n\n🔍 AI-RISK ASSESSMENT (High/Medium/Low) with reasons\n📝 SUSPICIOUS PHRASES\n✅ LIKELY ORIGINAL ELEMENTS\n💬 CONVERSATION SCRIPT for teacher\n📋 NEXT STEPS\n⚠️ NOTE: Advisory only — not proof of AI use.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/assessment_prep', methods=['POST'])
def do_assessment_prep():
    try:
        d, err, code = _get_json(["grade", "subject", "qtype", "num"])
        if err:
            return err, code
        state = d.get('state', 'worldwide')
        state_data = STATE_DATA.get(state, STATE_DATA['worldwide'])
        standard = d.get('standard', '').strip()
        std = standard if standard else "appropriate grade-level standard"
        assessment_name = state_data['assessment_name']
        return jsonify(result=llm(
            state_data['assessment_system_prompt'],
            f"Create {d['num']} {d['qtype']} {assessment_name} practice questions. Grade: {d['grade']} | Subject: {d['subject']} | Standard: {std}\n\nFor each:\n❓ QUESTION\n🅐 Answer choices (if MC)\n✅ CORRECT ANSWER\n💡 EXPLANATION\n📋 STANDARD ALIGNMENT",
            state,
        ))
    except Exception:
        return _internal_error()


@app.route('/parent_letter', methods=['POST'])
def do_parent():
    try:
        d, err, code = _get_json(["type", "lang", "details", "teacher"])
        if err:
            return err, code
        return jsonify(result=llm(
            f"Professional educator writing parent communications in {d['lang']}. Culturally responsive tone.",
            f"Write a {d['type']} parent letter in {d['lang']}.\nDetails: {d['details']}\nFrom: {d['teacher']}\nInclude: Date, greeting, clear explanation, action needed, contact info, closing. Under 250 words. Write ONLY in {d['lang']}.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/unit_plan', methods=['POST'])
def do_unit():
    try:
        d, err, code = _get_json(["topic", "grade", "subject", "duration"])
        if err:
            return err, code
        standard = d.get('standard', '').strip()
        std = standard if standard else "appropriate grade-level standard"
        return jsonify(result=llm(
            f"Expert {d['grade']} {d['subject']} curriculum designer.",
            f"Create a 2-week unit plan for: {d['topic']}\nStandard: {std} | Grade: {d['grade']} | Subject: {d['subject']} | Duration: {d['duration']}\n\n📌 UNIT OVERVIEW\n📅 WEEK 1 (Day 1–5)\n📅 WEEK 2 (Day 6–10)\n📊 ASSESSMENTS\n♿ DIFFERENTIATION\n📚 RESOURCES",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/iep504', methods=['POST'])
def do_iep504():
    try:
        d, err, code = _get_json(["plan", "grade", "disability", "context"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Special education expert. Knowledge of IDEA and Section 504.",
            f"Request: {d['plan']} | Grade: {d['grade']} | Need: {d['disability']} | Context: {d['context']}\n\n📋 PLAIN LANGUAGE EXPLANATION\n⚖️ LEGAL BASIS\n✅ SPECIFIC ACCOMMODATIONS (at least 8)\n🎯 CLASSROOM STRATEGIES\n👨‍👩‍👧 PARENT COMMUNICATION TIPS\n⚠️ Always consult your Child Study Team for official plans.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/quiz', methods=['POST'])
def do_quiz():
    try:
        d, err, code = _get_json(["topic", "grade", "qtype", "num"])
        if err:
            return err, code
        return jsonify(result=llm(
            "K-12 assessment specialist.",
            f"Create a {d['qtype']} quiz with {d['num']} questions for grade {d['grade']} on topic: {d['topic']}. Include answer key and short rationale.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/rubric', methods=['POST'])
def do_rubric():
    try:
        d, err, code = _get_json(["assignment", "grade", "criteria", "scale"])
        if err:
            return err, code
        return jsonify(result=llm(
            "K-12 instructional coach and rubric designer.",
            f"Build a {d['scale']} rubric for grade {d['grade']} assignment: {d['assignment']}. Include {d['criteria']} criteria with clear performance descriptors.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


@app.route('/refine_response', methods=['POST'])
def do_refine_response():
    try:
        d, err, code = _get_json(["response", "request"])
        if err:
            return err, code
        return jsonify(result=llm(
            "Instructional writing coach for teachers.",
            f"Revise the following AI response based on teacher request.\n\nORIGINAL RESPONSE:\n{d['response']}\n\nTEACHER REQUEST:\n{d['request']}\n\nReturn improved version plus a brief list of what changed.",
            d.get('state', 'worldwide'),
        ))
    except Exception:
        return _internal_error()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
