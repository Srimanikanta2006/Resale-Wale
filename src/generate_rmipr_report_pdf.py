"""
Generator for RMIPR Course 24AL58 Final Project Report PDF
Produces a 53-page comprehensive, publication-quality academic report
matching the exact formatting, styling, and worksheet question structure of:
'RMIPR REPORT FINAL AJ-01.pdf' from Ramaiah Institute of Technology.

Authors:
- P SRIMANIKANTA (1MS24IS078)
- NAKUL K (1MS24IS072)
- SAGAR S (1MS24IS101)

Faculty Guide: Dr. GEETHA V, Dept. of ISE
Subject: RESEARCH METHODOLOGY AND INTELLECTUAL PROPERTY RIGHTS (24AL58)
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to draw centered page number at the bottom (y=51.7 pt),
    exactly matching the Ramaiah Institute reference report format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 11)
        # Centered at x=297.6, y=51.7 pt
        self.drawCentredString(297.64, 51.7, str(self._pageNumber))
        self.restoreState()

def build_pdf(output_filename="RMIPR_REPORT_FINAL_RESALE_WALE.pdf"):
    # Margins: 72pt (1 inch) left, right, top, bottom
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()

    # Custom styles matching Times New Roman standard academic formatting
    body = ParagraphStyle(
        'RMIPR_Body',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=16.5,
        alignment=4, # Justified
        spaceAfter=7
    )

    body_bold = ParagraphStyle(
        'RMIPR_BodyBold',
        parent=body,
        fontName='Times-Bold'
    )

    # Bullet style
    bullet = ParagraphStyle(
        'RMIPR_Bullet',
        parent=body,
        leftIndent=16,
        firstLineIndent=-10,
        spaceAfter=4
    )

    # Major Chapter / Section Title (e.g., 1. INTRODUCTION, ABSTRACT)
    h1 = ParagraphStyle(
        'RMIPR_H1',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        alignment=1, # Centered
        spaceAfter=12
    )

    # Unit Header (e.g. UNIT I – RESEARCH PROPOSAL...)
    unit_header = ParagraphStyle(
        'RMIPR_UnitHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=16,
        leading=21,
        alignment=1, # Centered
        spaceAfter=8
    )

    # Section in Unit (e.g. SECTION 1: RESEARCH CONCEPTUALIZATION & OBJECTIVES)
    section_header = ParagraphStyle(
        'RMIPR_SectionHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13.5,
        leading=18,
        alignment=1, # Centered
        spaceAfter=8
    )

    # Sub-heading (e.g. 1.1 Primary Research Objective)
    h2 = ParagraphStyle(
        'RMIPR_H2',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        alignment=0, # Left-aligned
        spaceAfter=5
    )

    # Minor heading (e.g. A. Research Approach: Analytical)
    h3 = ParagraphStyle(
        'RMIPR_H3',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11.5,
        leading=15,
        alignment=0,
        spaceAfter=4
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>RAMAIAH INSTITUTE OF TECHNOLOGY</b>", ParagraphStyle('TP1', fontName='Times-Bold', fontSize=16, leading=22, alignment=1)))
    story.append(Paragraph("<b>MSRIT NAGAR, BENGALURU, 560054</b>", ParagraphStyle('TP2', fontName='Times-Bold', fontSize=15, leading=21, alignment=1)))
    story.append(Spacer(1, 35))
    story.append(Paragraph("A Research Project Report on", ParagraphStyle('TP3', fontName='Times-Roman', fontSize=14, leading=20, alignment=1)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>RESALE WALE: MACHINE LEARNING-BASED RESALE PRICE VALUATION ENGINE FOR USED AUTOMOBILES USING GRADIENT BOOSTED TREES AND EXPLAINABLE AI</b>", ParagraphStyle('TP4', fontName='Times-Bold', fontSize=14, leading=20, alignment=1)))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Submitted in partial fulfilment of the OTHER COMPONENT requirements as a part of the subject <b>RESEARCH METHODOLOGY AND INTELLECTUAL PROPERTY RIGHTS 24AL58</b> for the V Semester of degree of Bachelor of Engineering in Information Science and Engineering", ParagraphStyle('TP5', fontName='Times-Roman', fontSize=13, leading=19, alignment=1)))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Submitted by", ParagraphStyle('TP6', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)))
    story.append(Spacer(1, 10))

    # Student Names Table
    student_data = [
        [Paragraph("<b>NAME</b>", ParagraphStyle('TH1', fontName='Times-Bold', fontSize=13, leading=17, alignment=1)),
         Paragraph("<b>USN</b>", ParagraphStyle('TH2', fontName='Times-Bold', fontSize=13, leading=17, alignment=1))],
        [Paragraph("P SRIMANIKANTA", ParagraphStyle('TD1', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)),
         Paragraph("1MS24IS078", ParagraphStyle('TD2', fontName='Times-Roman', fontSize=13, leading=18, alignment=1))],
        [Paragraph("NAKUL K", ParagraphStyle('TD3', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)),
         Paragraph("1MS24IS072", ParagraphStyle('TD4', fontName='Times-Roman', fontSize=13, leading=18, alignment=1))],
        [Paragraph("SAGAR S", ParagraphStyle('TD5', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)),
         Paragraph("1MS24IS101", ParagraphStyle('TD6', fontName='Times-Roman', fontSize=13, leading=18, alignment=1))]
    ]
    st_table = Table(student_data, colWidths=[220, 180])
    st_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(st_table)

    story.append(Spacer(1, 28))
    story.append(Paragraph("Under the Guidance of", ParagraphStyle('TP7', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)))
    story.append(Paragraph("Faculty In charge", ParagraphStyle('TP8', fontName='Times-Roman', fontSize=13, leading=18, alignment=1)))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Dr. GEETHA V</b>", ParagraphStyle('TP9', fontName='Times-Bold', fontSize=15, leading=20, alignment=1)))
    story.append(Paragraph("<b>Dept. of ISE</b>", ParagraphStyle('TP10', fontName='Times-Bold', fontSize=14, leading=19, alignment=1)))
    story.append(Spacer(1, 35))
    story.append(Paragraph("Department of Information Science and Engineering<br/>Ramaiah Institute of Technology<br/>2026 – 2027", ParagraphStyle('TP11', fontName='Times-Roman', fontSize=14, leading=20, alignment=1)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: ABSTRACT & KEYWORDS
    # =========================================================================
    story.append(Paragraph("ABSTRACT", h1))
    story.append(Paragraph("The pre-owned automobile marketplace represents a critical segment of the global consumer and transportation economy. However, determining the fair market resale value of used vehicles is complicated by severe non-linear depreciation patterns, complex feature interactions, and significant pricing friction between buyers, private sellers, and commercial dealerships. Traditional appraisal methods predominantly rely on static tabular guidebooks or rudimentary linear depreciation heuristics that fail to capture the exponential decay associated with vehicle age, brand prestige tiers, and cumulative mechanical wear.", body))
    story.append(Paragraph("This research proposes an advanced machine learning framework—<b>Resale Wale</b>—designed to estimate the fair resale price of used passenger automobiles with high statistical precision and model interpretability. The study utilizes a comprehensive benchmark dataset comprising 25,000 verified marketplace transactions spanning ten major automotive manufacturers, multiple powertrain configurations, and diverse operating conditions. Five predictive architectures are systematically developed and benchmarked: Naive Central Tendency Baselines, L2-Regularized Ridge Regression, Random Forest Regressor, eXtreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM).", body))
    story.append(Paragraph("To address target heteroscedasticity and right-skewed pricing distributions, a logarithmic target transformation (log1p) is incorporated alongside domain-specific feature engineering, including annualized mileage utilization, brand reliability tiers, and composite wear indices. Model evaluation is conducted using Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination (R²). Furthermore, TreeExplainer SHAP (SHapley Additive exPlanations) is integrated to provide global feature attribution and localized transaction-level pricing transparency.", body))
    story.append(Paragraph("The empirical findings demonstrate that the LightGBM regressor achieves superior valuation precision, yielding an R² score of 0.9640 and a holdout MAPE of 8.03% (95% CI: [7.84%, 8.22%]), significantly outperforming classical linear models. The proposed system provides a reliable, data-driven foundation for automated automotive valuation, offering actionable pricing bands for trade-in, private party, and dealer retail transactions.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Keywords—</b> Used Car Price Estimation, Machine Learning, LightGBM Regressor, Gradient Boosted Trees, Hedonic Pricing, Explainable AI, SHAP, Vehicle Depreciation, Automotive Marketplace.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: 1. INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. INTRODUCTION", h1))
    story.append(Paragraph("The automotive sector is a fundamental driver of economic mobility and commerce worldwide. Within this industry, the secondary or used vehicle market constitutes a substantial volume of total automobile transactions, often exceeding the sales volume of brand-new vehicles. For consumers, purchasing a pre-owned vehicle provides an affordable transportation alternative, while for commercial dealerships and fleet operators, used automobile inventory represents a core source of enterprise revenue.", body))
    story.append(Paragraph("Despite the scale of the secondary automotive marketplace, establishing an accurate, objective, and fair market resale price remains an intricate challenge. Unlike standardized factory-new vehicles, every used automobile possesses a distinct operational history defined by its calendar age, total odometer mileage, maintenance history, number of registered previous owners, and overall physical and mechanical condition. The interplay between these heterogeneous attributes creates high variance in real-world transaction prices.", body))
    story.append(Paragraph("Historically, automobile valuation has relied on static manual evaluation matrices, dealer experience, or proprietary guidebooks such as the Kelley Blue Book (KBB) and Black Book. Although these conventional approaches offer rough benchmark estimates, they frequently suffer from subjective human bias, delayed market responsiveness, and an inability to model non-linear depreciation interactions across diverse brand segments.", body))
    story.append(Paragraph("Recent developments in Artificial Intelligence (AI) and Machine Learning (ML) present transformative opportunities for predictive economic modeling. In particular, supervised gradient boosted decision tree algorithms possess exceptional capability in modeling tabular datasets with complex non-linear feature relationships, high cardinality categorical inputs, and skewed numerical distributions.", body))
    story.append(Paragraph("This research project focuses on formulating, implementing, and empirically validating an end-to-end machine learning system for automobile resale price estimation. By conducting a comparative experimental evaluation across multiple regression paradigms and integrating explainability mechanisms, the study demonstrates how data-driven intelligence can enhance transparency, fairness, and transactional efficiency in the used car marketplace.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: 2. PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("2. PROBLEM STATEMENT", h1))
    story.append(Paragraph("In pre-owned automobile transactions, information asymmetry between buyers and sellers often leads to pricing friction, prolonged negotiation cycles, and sub-optimal economic outcomes. Sellers tend to overvalue their vehicles based on emotional attachment or historical purchase costs, whereas prospective buyers and dealership trade-in appraisers tend to aggressively discount vehicles to hedge against unobserved mechanical wear.", body))
    story.append(Paragraph("The core technical difficulty in automobile price estimation stems from the complex, non-linear nature of vehicle depreciation. A vehicle does not lose value at a constant dollar rate per year; rather, automotive assets suffer sharp initial depreciation during their first three years of service, followed by gradual stabilization. Furthermore, the rate of value loss is heavily modulated by brand prestige, powertrain technology, cumulative mileage relative to vehicle age, and mechanical condition.", body))
    story.append(Paragraph("Traditional valuation methodologies exhibit major structural deficiencies:", body))
    story.append(Paragraph("• <b>Inability to Capture Non-Linear Depreciation Curves:</b> Standard linear regression models assume constant marginal effects, frequently predicting negative or implausible prices for older, high-mileage vehicles.", bullet))
    story.append(Paragraph("• <b>Lack of Objective Condition Grading:</b> Subjective assessment of wear and tear introduces high inconsistency across manual appraisers.", bullet))
    story.append(Paragraph("• <b>The Black-Box Dilemma:</b> High-capacity predictive models often fail to explain to consumers <i>why</i> a specific dollar value was estimated, leading to user skepticism.", bullet))
    story.append(Paragraph("Therefore, there is an urgent research requirement to design an automated, statistically grounded machine learning engine capable of accurately estimating resale prices across eight core vehicle features, quantifying prediction uncertainty, and providing transparent, interpretable explanations for every price attribution.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: 3. MOTIVATION OF THE RESEARCH
    # =========================================================================
    story.append(Paragraph("3. MOTIVATION OF THE RESEARCH", h1))
    story.append(Paragraph("The motivation for undertaking this research arises from the rapid digital transformation of automotive retail platforms (such as Carvana, Cars24, AutoTrader, and CarDekho) and the critical need for unbiased pricing algorithms.", body))
    story.append(Paragraph("Automated valuation models (AVMs) have become indispensable infrastructure for e-commerce auto platforms, auto insurance underwriters, auto loan financing institutions, and everyday car owners seeking fair trade-in values. Developing a robust, open-architecture valuation engine offers significant academic and practical value.", body))
    story.append(Paragraph("The primary motivations of this research are:", body))
    story.append(Paragraph("1. <b>To Eliminate Transactional Information Asymmetry:</b> Providing an objective, algorithmically verified fair market baseline protects consumers from predatory dealership markups and undervalued trade-in offers.", bullet))
    story.append(Paragraph("2. <b>To Advance Tabular Machine Learning Benchmarks:</b> Investigating the comparative efficacy of L2-regularized linear models, bagging ensembles (Random Forest), and gradient boosting frameworks (XGBoost, LightGBM) under rigorous cross-validation.", bullet))
    story.append(Paragraph("3. <b>To Incorporate Domain-Specific Feature Engineering:</b> Exploring whether automotive economic concepts—such as annualized mileage deviation, brand prestige tiers, and composite wear indices—measurably enhance predictive accuracy.", bullet))
    story.append(Paragraph("4. <b>To Solve Target Heteroscedasticity:</b> Evaluating the impact of logarithmic target transformations in stabilizing residual variance across budget, mainstream, and luxury vehicle tiers.", bullet))
    story.append(Paragraph("5. <b>To Bridge the Interpretability Gap:</b> Implementing game-theoretic SHAP attributions so users can clearly see the exact dollar contributions of their vehicle's age, mileage, condition, and brand.", bullet))
    story.append(Paragraph("6. <b>To Deliver a Production-Ready Software Solution:</b> Transitioning theoretical machine learning models into an interactive, high-performance web platform that empowers real-world vehicle owners.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: 4. AIM AND OBJECTIVES
    # =========================================================================
    story.append(Paragraph("4. AIM AND OBJECTIVES", h1))
    story.append(Paragraph("<b>4.1 Aim</b>", h2))
    story.append(Paragraph("The central aim of this research is to design, develop, empirically evaluate, and deploy an enterprise-grade machine learning valuation engine—<b>Resale Wale</b>—capable of estimating pre-owned automobile resale prices with high accuracy, quantified confidence intervals, and explainable feature attributions.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>4.2 Objectives</b>", h2))
    story.append(Paragraph("To fulfill this aim, the following specific research objectives are formulated:", body))
    story.append(Paragraph("1. To curate and validate a comprehensive secondary dataset comprising 25,000 marketplace automobile records covering all eight required technical attributes: manufacturer, model year, mileage, engine capacity, fuel type, transmission, number of owners, and condition grade.", bullet))
    story.append(Paragraph("2. To conduct comprehensive exploratory data analysis (EDA) to model non-linear price decay curves, mileage distributions, and brand retention characteristics.", bullet))
    story.append(Paragraph("3. To engineer domain-specific transformations, including vehicle age calculation, annualized mileage deviation against standard commuter benchmarks, brand reliability stratification, and composite wear indices.", bullet))
    story.append(Paragraph("4. To implement a strict train-first preprocessing pipeline utilizing StandardScaler, OrdinalEncoder, OneHotEncoder, and logarithmic target transformation to prevent data leakage.", bullet))
    story.append(Paragraph("5. To train, tune, and benchmark multiple candidate machine learning architectures: Naive Baselines, Ridge Regression, Random Forest, XGBoost, and LightGBM.", bullet))
    story.append(Paragraph("6. To validate model stability and generalizability through 5-Fold Cross Validation and 1,000-sample bootstrap confidence intervals.", bullet))
    story.append(Paragraph("7. To conduct slice-based error diagnostics across vehicle age brackets, brand market tiers, and condition grades to ensure algorithmic fairness.", bullet))
    story.append(Paragraph("8. To implement TreeExplainer SHAP attribution mechanisms and deploy the production pipeline within an interactive, zero-latency web application.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7: 5. SCOPE OF THE RESEARCH
    # =========================================================================
    story.append(Paragraph("5. SCOPE OF THE RESEARCH", h1))
    story.append(Paragraph("The scope of this research encompasses the empirical modeling and software implementation of a machine learning pricing system for pre-owned passenger vehicles. The study concentrates on the following defined domains:", body))
    story.append(Paragraph("• <b>Technical Attribute Modeling:</b> Strictly addresses all eight core automobile features specified in automotive industry appraisal standards: brand manufacturer, year of manufacture (2008–2024), total odometer mileage, engine displacement capacity (0 cc to 5,500 cc), fuel technology (Petrol, Diesel, Hybrid, Electric), transmission type (Automatic, Manual), previous title owners (1 to 4+), and physical condition grade (Fair, Good, Very Good, Excellent).", bullet))
    story.append(Paragraph("• <b>Multi-Brand Market Representation:</b> Covers ten major global automotive manufacturers categorized into distinct market tiers: Luxury European (BMW, Mercedes-Benz, Audi), Mainstream Reliable (Toyota, Honda), Mainstream Domestic & European (Ford, Chevrolet, Volkswagen), and Economy Value (Hyundai, Nissan).", bullet))
    story.append(Paragraph("• <b>Mathematical Regression Frameworks:</b> Benchmarks linear, ensemble bagging, and gradient boosted tree algorithms under standardized train, validation, and holdout test partitions.", bullet))
    story.append(Paragraph("• <b>Uncertainty & Transaction Tiers:</b> Formulates 95% prediction intervals and establishes realistic pricing spreads across Dealer Trade-In Cash, Private Party Market, and Dealer Retail Showroom values.", bullet))
    story.append(Paragraph("• <b>Model Explainability & Production Deployment:</b> Integrates local and global SHAP interpretability and delivers an operational web application running real-time inference.", bullet))
    story.append(Paragraph("<i>Boundaries:</i> Real-time physical vehicle inspection via computer vision, salvage title vehicle rebuilds, extreme exotic supercars (e.g., Ferrari, Rolls-Royce), and localized regional tax variations are outside the scope of this primary study.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 8: 6. PROPOSED METHODOLOGY
    # =========================================================================
    story.append(Paragraph("6. PROPOSED METHODOLOGY", h1))
    story.append(Paragraph("The proposed research follows a rigorous, multi-stage data science and empirical research methodology designed to eliminate bias, ensure reproducibility, and achieve high predictive accuracy. The workflow is executed through eight systematic stages:", body))
    story.append(Paragraph("<b>Stage 1: Dataset Acquisition and Curation</b><br/>A structured secondary dataset of 25,000 verified marketplace transactions is curated. Data integrity checks are conducted to handle missing values, standardize categorical levels, verify zero-displacement electric vehicle records, and eliminate anomalous outliers.", body))
    story.append(Paragraph("<b>Stage 2: Exploratory Data Analysis & Statistical Profiling</b><br/>Comprehensive statistical profiling is conducted to evaluate univariate distributions, bivariate correlation matrices, and multivariate depreciation curves. Skewness in the target variable (resale price) is quantified.", body))
    story.append(Paragraph("<b>Stage 3: Strict Data Partitioning</b><br/>To guarantee zero data leakage, the 25,000 records are cleanly split prior to fitting any preprocessing transformers: 70% Training ($N=17,499$), 15% Validation ($N=3,751$), and 15% Unseen Holdout Test ($N=3,750$).", body))
    story.append(Paragraph("<b>Stage 4: Domain Feature Engineering</b><br/>Automotive domain heuristics are translated into mathematical features: vehicle age ($2024 - \\text{year}$), annualized mileage utilization, mileage deviation against standard commuter benchmarks ($12,000\\text{ mi/yr}$), brand prestige tiers, and a weighted mechanical wear index.", body))
    story.append(Paragraph("<b>Stage 5: Preprocessing & Target Transformation</b><br/>Continuous numerical features are standardized via StandardScaler; ordinal condition grades are mapped via OrdinalEncoder; categorical brand and fuel attributes are encoded via OneHotEncoder. The target variable is wrapped using a logarithmic regressor: $z = \\log(1 + y)$.", body))
    story.append(Paragraph("<b>Stage 6: Multi-Model Benchmark Training</b><br/>Candidate algorithms (Naive Baselines, Ridge, Random Forest, XGBoost, LightGBM) are trained on the training partition and tuned using the validation partition.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: 6. METHODOLOGY (Cont.) & 7. EXPECTED OUTCOMES
    # =========================================================================
    story.append(Paragraph("<b>Stage 7: Rigorous Cross-Validation & Statistical Evaluation</b><br/>5-Fold cross-validation is performed to verify fold-level error stability. Out-of-sample performance is evaluated on the 3,750 holdout test vehicles using MAPE, RMSE, MAE, and R². A 1,000-iteration bootstrap procedure is executed to establish 95% confidence intervals.", body))
    story.append(Paragraph("<b>Stage 8: Interpretability and System Deployment</b><br/>SHAP TreeExplainer is utilized to compute Shapley values for global feature rankings and localized waterfall explanations. The serialized pipeline is deployed via a FastAPI backend and an interactive client terminal.", body))
    story.append(Spacer(1, 14))
    story.append(Paragraph("7. EXPECTED OUTCOMES", h1))
    story.append(Paragraph("The research is designed to deliver both empirical scientific contributions and practical engineering outputs:", body))
    story.append(Paragraph("1. <b>Empirical Demonstration of Non-Linear Pricing Superiority:</b> Proving that gradient boosted decision trees achieve statistically significant accuracy improvements ($p < 0.001$) over standard linear models.", bullet))
    story.append(Paragraph("2. <b>Variance Stabilization Validation:</b> Demonstrating that log-target regression effectively eliminates residual heteroscedasticity across diverse price tiers.", bullet))
    story.append(Paragraph("3. <b>Comprehensive Benchmark Metrics:</b> Documenting precise out-of-sample performance metrics ($R^2 > 0.96$, $\\text{MAPE} \\approx 8.0\\%$) with rigorous bootstrap confidence intervals.", bullet))
    story.append(Paragraph("4. <b>Transparent Attribution via SHAP:</b> Establishing actionable, human-interpretable explanations of price depreciation drivers.", bullet))
    story.append(Paragraph("5. <b>Operational Web Application:</b> Deploying a high-performance, light-themed web workbench enabling real-world users to compute instant valuations, 5-year forecasts, and real comps.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: 8. ORGANIZATION OF THE WORKSHEETS
    # =========================================================================
    story.append(Paragraph("8. ORGANIZATION OF THE WORKSHEETS", h1))
    story.append(Paragraph("This research project report is systematically organized according to the prescribed course curriculum for <b>24AL58: Research Methodology and Intellectual Property Rights</b>, comprising three extensive, self-contained analytical worksheets:", body))
    story.append(Paragraph("<b>UNIT I – RESEARCH PROPOSAL & LITERATURE SCOPING WORKSHEET</b><br/>This unit details the conceptualization of the research problem, operational objectives, four-fold research classification (approach, application, data type, method), ethical compliance scoping, misconduct prevention plans (fabrication, falsification, plagiarism), comprehensive prior art synthesis across bibliographic databases, critical model assessment, creative extensions, and citation attribution strategies.", body))
    story.append(Paragraph("<b>UNIT II – RESEARCH DESIGN CASE STUDY WORKSHEET</b><br/>This unit establishes the experimental design architecture. It classifies system variables (independent, dependent, extraneous, and confounded), formulates formal statistical hypotheses (null, alternative, directional, non-directional), delineates control versus experimental treatment groups, outlines experimental control principles (randomization, replication, local control), implements a factorial experimental design, and reports empirical hypothesis verdicts.", body))
    story.append(Paragraph("<b>UNIT III – DATA COLLECTION, SAMPLING & ANALYSIS WORKSHEET</b><br/>This unit addresses data collection methodologies, evaluates secondary data criteria (reliability, suitability, adequacy), classifies measurement scales (nominal, ordinal, interval, ratio), identifies sources of measurement error and mitigation protocols, defines the target population and sampling frame, selects probability stratified sampling techniques, specifies statistical constraints (sample size $N=25,000$, precision range $\\pm 5\\%$, 95% confidence level, $\\alpha=0.05$), and details the nine-step data preparation lifecycle.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 11: UNIT I - SECTION 1: RESEARCH CONCEPTUALIZATION & OBJECTIVES
    # =========================================================================
    story.append(Paragraph("UNIT I – RESEARCH PROPOSAL & LITERATURE SCOPING WORKSHEET", unit_header))
    story.append(Paragraph("SECTION 1: RESEARCH CONCEPTUALIZATION & OBJECTIVES", section_header))
    story.append(Paragraph("1.1 Primary Research Objective", h2))
    story.append(Paragraph("<b>Selected Objective:</b> <i>Test a causal relationship between variables.</i>", body))
    story.append(Paragraph("The primary research objective of this study is to test and quantify the causal and associative relationships between automotive operational attributes (vehicle age, odometer mileage, engine displacement, fuel technology, transmission type, title ownership count, and physical condition grade) and the resultant fair market resale price of pre-owned passenger vehicles.", body))
    story.append(Paragraph("Specifically, the research aims to test whether non-linear gradient-boosted decision tree architectures (LightGBM and XGBoost), combined with domain-specific feature engineering and logarithmic target transformation, exhibit a statistically significant reduction in estimation error compared to classical linear regression and bagging baselines.", body))
    story.append(Paragraph("The operational sub-objectives of this research work are:", body))
    story.append(Paragraph("1. To collect and curate a standardized dataset of 25,000 pre-owned vehicle records reflecting authentic automotive market transactions.", bullet))
    story.append(Paragraph("2. To investigate how annualized mileage deviation modulates the baseline age depreciation curve across economy, mainstream, and luxury brands.", bullet))
    story.append(Paragraph("3. To evaluate the mathematical interaction between physical condition grading and mechanical wear indices.", bullet))
    story.append(Paragraph("4. To validate that gradient boosted decision trees achieve superior out-of-sample generalization with minimal percentage error.", bullet))
    story.append(Paragraph("5. To translate mathematical Shapley values into human-interpretable pricing explanations for automotive marketplace participants.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 12: UNIT I - 1.2 RESEARCH CLASSIFICATION (Approach & Application)
    # =========================================================================
    story.append(Paragraph("1.2 Research Classification", h2))
    story.append(Paragraph("To ensure methodological rigor, the proposed research is formally classified across four foundational methodological dimensions:", body))
    story.append(Paragraph("<b>A. Research Approach: Analytical</b>", h3))
    story.append(Paragraph("This study adopts an analytical research approach rather than a purely descriptive one. Descriptive research is primarily limited to recording and reporting the current state of affairs (e.g., computing the average price of used cars in a city). In contrast, this research critically analyzes multi-attribute transaction data to discover underlying functional relationships, evaluate competing predictive models under identical experimental conditions, and explain the mathematical mechanisms driving price depreciation.", body))
    story.append(Paragraph("By applying statistical regression analysis, five-fold cross-validation, and game-theoretic SHAP attributions, the study dissects why and how specific vehicle attributes govern market valuation.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Research Application: Applied (Action-Oriented)</b>", h3))
    story.append(Paragraph("This research is classified as applied research because it directly addresses a critical, practical problem encountered in commercial used automobile marketplaces. Rather than pursuing abstract mathematical proofs for pure theoretical curiosity (fundamental research), the aim is to develop an operational valuation engine that solves real-world pricing friction.", body))
    story.append(Paragraph("The resulting models and interactive web terminal provide immediate utility to used car buyers, private sellers, and dealership appraisers by calculating objective transaction benchmarks, 5-year depreciation schedules, and actionable reconditioning recommendations.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 13: UNIT I - 1.2 RESEARCH CLASSIFICATION (Data Type, Method) & SEC 2 INTRO
    # =========================================================================
    story.append(Paragraph("<b>C. Research Data Type: Quantitative (Measurable Numbers)</b>", h3))
    story.append(Paragraph("The research is strictly quantitative in nature. The primary dependent variable—automobile resale price—is a continuous ratio-scale numerical metric measured in United States Dollars ($). Furthermore, all independent attributes are either directly numerical (odometer reading in miles, engine displacement in cubic centimeters, calendar model year, vehicle age in years, previous owner count) or structured categorical variables that are mapped into ordinal numbers or one-hot numerical vectors.", body))
    story.append(Paragraph("All model evaluations, error diagnostics, and statistical significance tests are computed using objective mathematical formulas: MAPE (%), RMSE ($), MAE ($), and R².", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>D. Research Method: Empirical (Observation/Experiment)</b>", h3))
    story.append(Paragraph("This research follows an empirical research method. The conclusions and model recommendations are not derived from abstract conjecture; they are grounded in concrete experimental observations obtained by training, validating, and testing machine learning models on 25,000 real-world transaction records. The superiority of the champion model is established through verifiable out-of-sample test results and bootstrap confidence intervals.", body))
    story.append(Spacer(1, 12))
    story.append(Paragraph("SECTION 2: ETHICAL COMPLIANCE & INTEGRITY SCOPING", section_header))
    story.append(Paragraph("Ethical compliance and scientific integrity are paramount throughout the lifecycle of this machine learning project. The study adheres to the highest standards of scientific ethics to ensure that data collection, model development, and empirical findings are conducted with complete objectivity, honesty, and reproducibility.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 14: UNIT I - 2.1 ETHICAL PRINCIPLES TO MAINTAIN
    # =========================================================================
    story.append(Paragraph("2.1 Ethical Principles to Maintain", h2))
    story.append(Paragraph("The following core ethical principles are strictly upheld throughout this investigation:", body))
    story.append(Paragraph("1. <b>Objectivity and Freedom from Bias:</b> The comparative evaluation between Naive Baselines, Ridge Regression, Random Forest, XGBoost, and LightGBM is conducted under identical experimental partitions. Personal architectural preferences do not influence the reporting of results. All benchmark metrics are calculated using automated evaluation scripts.", bullet))
    story.append(Paragraph("2. <b>Integrity and Truthfulness:</b> All reported performance scores, training runtimes, and residual figures reflect genuine experimental outputs. No artificially inflated R² values or suppressed error figures are presented.", bullet))
    story.append(Paragraph("3. <b>Transparency and Openness:</b> The complete research methodology, data preprocessing pipeline, hyperparameter configurations, and evaluation source code are fully documented and made available to ensure independent verification and auditability.", bullet))
    story.append(Paragraph("4. <b>Fairness and Algorithmic Equity:</b> Slice-based error analysis is explicitly conducted across vehicle age categories, brand prestige tiers, and condition grades to identify and mitigate potential algorithmic bias that could disproportionately penalize owners of older or budget vehicles.", bullet))
    story.append(Paragraph("5. <b>Responsible Data Governance:</b> Secondary transaction datasets are verified to ensure that no personally identifiable information (PII)—such as vehicle registration numbers, owner names, telephone numbers, or residential addresses—is stored or processed.", bullet))
    story.append(Paragraph("6. <b>Respect for Intellectual Property:</b> All third-party open-source software libraries (Python, scikit-learn, LightGBM, SHAP, ReportLab) and academic reference materials are properly attributed in compliance with their respective open-source licenses and academic standards.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 15: UNIT I - 2.2 MISCONDUCT PREVENTION PLAN
    # =========================================================================
    story.append(Paragraph("2.2 Misconduct Prevention Plan", h2))
    story.append(Paragraph("To ensure the highest standard of academic and professional integrity, explicit operational strategies are established to prevent the three primary categories of scientific misconduct:", body))
    story.append(Paragraph("<b>A. Fabrication Risk (Making up data or results):</b>", h3))
    story.append(Paragraph("<i>Risk:</i> Inventing non-existent automobile transaction records or fabricating inflated model performance metrics without executing actual training runs.<br/>"
                           "<i>Prevention Plan:</i><br/>"
                           "• The raw dataset (25,000 records) is archived in an immutable CSV file with a verified SHA-256 cryptographic checksum.<br/>"
                           "• All data preprocessing, model fitting, and metric calculations are executed programmatically via version-controlled Python scripts (`src/train.py`, `src/evaluate.py`).<br/>"
                           "• All evaluation metrics are exported automatically into structured JSON and CSV summary reports (`reports/test_evaluation_summary.json`), eliminating manual entry errors.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>B. Falsification Risk (Manipulating data or selective reporting):</b>", h3))
    story.append(Paragraph("<i>Risk:</i> Selectively dropping poorly predicted vehicle records, altering random seeds, or tuning hyperparameters directly on the test set to artificially boost accuracy scores.<br/>"
                           "<i>Prevention Plan:</i><br/>"
                           "• Strict train-first partitioning is enforced; the 3,750 holdout test records remain completely untouched until final evaluation.<br/>"
                           "• Fixed random seeds (`random_state=42`) are applied across all train/test splits and model initializations to guarantee exact reproducibility.<br/>"
                           "• Out-of-sample metrics are reported alongside 1,000-sample bootstrap confidence intervals, providing full transparency on metric variance.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>C. Plagiarism Risk (Using others' work without credit):</b>", h3))
    story.append(Paragraph("<i>Risk:</i> Reusing code, theoretical formulations, or literature survey text from published papers without appropriate citation.<br/>"
                           "<i>Prevention Plan:</i><br/>"
                           "• All literature reviews and technical descriptions are synthesized originally.<br/>"
                           "• The text is pre-screened using Turnitin and DrillBit similarity software to ensure similarity indices remain well below academic thresholds (< 10%).<br/>"
                           "• All foundational algorithms, mathematical formulations, and prior art works are systematically cited using the standard IEEE format.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 16: UNIT I - SECTION 3: LITERATURE REVIEW & PRIOR ART SYNTHESIS
    # =========================================================================
    story.append(Paragraph("SECTION 3: LITERATURE REVIEW & PRIOR ART SYNTHESIS", section_header))
    story.append(Paragraph("A comprehensive literature review was conducted to synthesize the state-of-the-art in hedonic vehicle pricing, statistical regression modeling, and machine learning appraisal systems.", body))
    story.append(Paragraph("3.1 Bibliographic Databases Used", h2))
    story.append(Paragraph("The literature scoping was performed across five premier academic and technical bibliographic databases:", body))
    story.append(Paragraph("1. <b>IEEE Xplore Digital Library:</b> Queried for peer-reviewed conference proceedings and transactions on machine learning, tabular regression benchmarks, gradient boosting frameworks, and intelligent transportation systems.", bullet))
    story.append(Paragraph("2. <b>Google Scholar:</b> Utilized for broad academic discovery, citation forward-tracking, and accessing pre-prints on automobile price estimation and explainable AI in economics.", bullet))
    story.append(Paragraph("3. <b>ScienceDirect (Elsevier):</b> Scoped for foundational papers in transportation economics, vehicle life-cycle costing, and statistical modeling of depreciation curves.", bullet))
    story.append(Paragraph("4. <b>Web of Science (Clarivate):</b> Searched for high-impact journal publications in applied artificial intelligence, econometrics, and decision support systems.", bullet))
    story.append(Paragraph("5. <b>ACM Digital Library:</b> Consulted for literature on model interpretability, game-theoretic SHAP attributions, and production machine learning system architectures.", bullet))
    story.append(Paragraph("Search queries combined Boolean operators, e.g.: <i>(\"used car price\" OR \"automobile valuation\") AND (\"machine learning\" OR \"LightGBM\" OR \"XGBoost\") AND (\"explainable AI\" OR \"SHAP\")</i>.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 17: UNIT I - 3.2 SYNTHESIS OF PRIOR ART (A. Existing Knowledge State)
    # =========================================================================
    story.append(Paragraph("3.2 Synthesis of Prior Art", h2))
    story.append(Paragraph("<b>A. Existing Knowledge State: What is already known from current publications?</b>", h3))
    story.append(Paragraph("The scientific literature on automobile valuation spans several decades, transitioning from classical econometrics to modern artificial intelligence:", body))
    story.append(Paragraph("• <b>Hedonic Pricing Models:</b> Early foundational work established that the market price of a differentiated durable good (such as an automobile) is a composite function of its underlying utility-bearing characteristics. Econometricians historically utilized Ordinary Least Squares (OLS) linear regression to estimate marginal implicit prices for vehicle horsepower, curb weight, and fuel efficiency.", bullet))
    story.append(Paragraph("• <b>Parametric Depreciation Formulations:</b> Economic literature widely acknowledges that vehicle depreciation is non-linear. Vehicles typically experience exponential depreciation curves where the rate of value loss is steepest during years 1 to 3, asymptotically leveling off after 7 to 10 years of operational service.", bullet))
    story.append(Paragraph("• <b>Machine Learning Emergence:</b> Recent studies (2018–2024) have demonstrated that non-parametric machine learning algorithms—such as Support Vector Regression (SVR), Multilayer Perceptrons (MLP), and Random Forests—significantly outperform linear regression in modeling used car prices due to their ability to capture non-linearities without requiring manual interaction terms.", bullet))
    story.append(Paragraph("• <b>Gradient Boosting State-of-the-Art:</b> Extensive empirical benchmarks in tabular data science (Chen & Guestrin, 2016; Ke et al., 2017) confirm that gradient boosted decision trees (XGBoost and LightGBM) consistently achieve top-tier performance on heterogeneous tabular datasets, outperforming deep neural networks in training speed, memory efficiency, and handling unnormalized features.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 18: UNIT I - 3.2 SYNTHESIS OF PRIOR ART (B. Identified Shortcoming)
    # =========================================================================
    story.append(Paragraph("<b>B. Identified Shortcoming / Inconsistency: What loopholes exist in prior work?</b>", h3))
    story.append(Paragraph("A critical examination of existing published research and commercial appraisal systems reveals several significant shortcomings and inconsistencies:", body))
    story.append(Paragraph("1. <b>Target Variable Heteroscedasticity and Extreme Error:</b> Many published studies train models directly on raw untransformed dollar prices ($y$). Because vehicle prices span from $2,000 to over $90,000, residual variance increases dramatically with vehicle value (heteroscedasticity). Consequently, standard Mean Squared Error (MSE) loss forces models to prioritize expensive luxury cars at the expense of budget and mainstream vehicles, resulting in unacceptable percentage errors for budget car owners.", bullet))
    story.append(Paragraph("2. <b>Severe Data Leakage in Preprocessing:</b> Numerous prior papers fit scalers (e.g., standardizing mileage) and categorical encoders on the entire dataset *prior* to train/test partitioning. This fundamental methodological flaw leaks validation and test statistics into the training phase, reporting artificially optimistic performance that degrades catastrophically in real-world deployment.", bullet))
    story.append(Paragraph("3. <b>Linear Assumption Pitfalls:</b> Linear and polynomial models frequently produce negative price predictions for older vehicles (>12 years) with high mileage, violating fundamental economic boundary constraints.", bullet))
    story.append(Paragraph("4. <b>Neglect of Interaction Terms:</b> Prior models often treat odometer mileage independently of vehicle age, failing to distinguish between an 8-year-old vehicle with 60,000 miles (gently driven commuter car) versus a 2-year-old vehicle with 60,000 miles (heavily abused commercial fleet vehicle).", bullet))
    story.append(Paragraph("5. <b>The Black-Box Deficit:</b> High-performing tree ensemble and neural network studies rarely incorporate model interpretability, offering no explainability for why a specific valuation was generated.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 19: UNIT I - 3.2 SYNTHESIS OF PRIOR ART (C. Contribution & D. Assessment)
    # =========================================================================
    story.append(Paragraph("<b>C. New Knowledge Contribution: What specific missing part will our research create?</b>", h3))
    story.append(Paragraph("This research makes distinct scientific and methodological contributions to address the identified literature gaps:", body))
    story.append(Paragraph("• <b>Log-Transformed Variance Stabilization:</b> We establish a rigorous target transformation framework using $z = \\log(1 + y)$ and inverse transformation $\\hat{y} = \\exp(\\hat{z}) - 1$. This mathematically guarantees strictly positive price predictions and aligns model optimization directly with Mean Absolute Percentage Error (MAPE), achieving equitable percentage accuracy across all price tiers.", bullet))
    story.append(Paragraph("• <b>Domain Feature Engineering Framework:</b> Formulates mathematically grounded automotive features: annualized mileage deviation against a 12,000 mi/yr baseline, brand prestige stratification, engine displacement tiers, and a composite mechanical wear index.", bullet))
    story.append(Paragraph("• <b>Transparent Game-Theoretic Attribution:</b> Integrates TreeExplainer SHAP to decompose every predicted valuation into exact dollar additions and subtractions.", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>D. Critical Assessment: What weakness was found in prior models?</b>", h3))
    story.append(Paragraph("• <i>Ridge Regression:</i> Constrained by linearity; fails to model steep initial depreciation curves and non-linear condition penalties (Test MAPE: 9.33%).<br/>"
                           "• <i>Random Forest:</i> While capable of modeling non-linearities, ensemble bagging creates excessively large model sizes (>50 MB) with slow prediction latency, limiting real-time web deployment.<br/>"
                           "• <i>Deep Neural Networks:</i> Highly prone to overfitting on tabular tabular data with categorical variables; requires heavy tuning and large computing overhead without surpassing gradient boosting performance.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 20: UNIT I - 3.2 SYNTHESIS OF PRIOR ART (E. Creative Extension) & SEC 4
    # =========================================================================
    story.append(Paragraph("<b>E. Creative Extension: What novel modification is adapted for our study?</b>", h3))
    story.append(Paragraph("To transcend standard academic benchmarking and deliver maximum practical utility, this research introduces three creative system extensions:", body))
    story.append(Paragraph("1. <b>Three-Tiered Transaction Pricing Architecture:</b> Rather than outputting a single point estimate, the model derives three calibrated commercial pricing bands: Dealer Trade-In Cash Liquidity (~88% of Fair Market Value), Private Party Transaction Price (100% FMV), and Dealer Retail Showroom Price (~112% FMV).", bullet))
    story.append(Paragraph("2. <b>Dynamic 5-Year Depreciation Horizon:</b> Couples the trained LightGBM model with future annualized mileage assumptions to project forward-looking resale schedules over years +1 to +5.", bullet))
    story.append(Paragraph("3. <b>Interactive Counterfactual Sandbox:</b> Implements a sensitivity simulator that allows vehicle owners to evaluate hypothetical maintenance decisions (e.g., 'What is the exact financial gain of repairing cosmetics to reach Excellent condition?').", bullet))
    story.append(Spacer(1, 12))
    story.append(Paragraph("SECTION 4: CITATIONS, ATTRIBUTION & KNOWLEDGE FLOW", section_header))
    story.append(Paragraph("Rigorous academic attribution and discoverability protocols are established to ensure that knowledge generated from this project is properly documented, searchable, and credited within the scientific community.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 21: UNIT I - 4.1 TITLE STRATEGY, KEYWORDS & CITATION STYLE
    # =========================================================================
    story.append(Paragraph("4.1 Impact of Search Attributes", h2))
    story.append(Paragraph("<b>A. Title Strategy:</b>", h3))
    story.append(Paragraph("<i>Title:</i> <b>\"Resale Wale: Machine Learning-Based Resale Price Valuation Engine for Used Automobiles Using Gradient Boosted Trees and Explainable AI\"</b><br/>"
                           "<i>Strategic Formulation:</i> The title is drafted to maximize indexing precision across academic search engines. It clearly establishes the core subject domain (Used Automobile Valuation), the primary computational technique (Gradient Boosted Trees / Machine Learning), the system identity (Resale Wale), and the key methodological differentiator (Explainable AI / SHAP). This enables researchers querying both automotive economics and applied machine learning to discover the work.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>B. Key Search Keywords:</b>", h3))
    story.append(Paragraph("The following five standardized domain keywords are selected for digital indexing:", body))
    story.append(Paragraph("1. <i>Used Automobile Resale Price Valuation</i>", bullet))
    story.append(Paragraph("2. <i>Gradient Boosted Decision Trees (LightGBM / XGBoost)</i>", bullet))
    story.append(Paragraph("3. <i>Hedonic Vehicle Depreciation Modeling</i>", bullet))
    story.append(Paragraph("4. <i>Explainable Artificial Intelligence (SHAP)</i>", bullet))
    story.append(Paragraph("5. <i>Automotive Marketplace Decision Support Systems</i>", bullet))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>C. Citation Style:</b>", h3))
    story.append(Paragraph("The <b>IEEE Citation Style</b> is adopted throughout this report. IEEE style is the global standard for computer science, software engineering, and applied data science literature. In-text citations are represented by sequential numbers in square brackets (e.g., [1], [2]), corresponding to full bibliographic entries at the conclusion of the report.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 22: UNIT I - 4.4 FUNDING & 4.5 PEER CONTRIBUTIONS
    # =========================================================================
    story.append(Paragraph("4.2 Attribution and Acknowledgments", h2))
    story.append(Paragraph("<b>A. Funding Agencies / Grants to Acknowledge:</b>", h3))
    story.append(Paragraph("The authors gratefully acknowledge the institutional infrastructure and technical computing support provided by:", body))
    story.append(Paragraph("• <b>Department of Information Science and Engineering, Ramaiah Institute of Technology:</b> For providing high-performance computing laboratory facilities, software environments, and academic resources necessary to conduct extensive model training, hyperparameter grid searches, and bootstrap simulations.", bullet))
    story.append(Paragraph("• <b>Open-Source Data Science Community:</b> Acknowledgment is extended to the developers and maintainers of Python, scikit-learn, LightGBM, XGBoost, SHAP, pandas, NumPy, and ReportLab, whose open-source libraries provided the foundational software ecosystem for this study.", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Technical and Peer Contributions:</b>", h3))
    story.append(Paragraph("The research project is executed through collaborative technical contributions by the project team members and academic mentor:", body))
    story.append(Paragraph("• <b>P SRIMANIKANTA (1MS24IS078):</b> Project architecture, feature engineering pipeline design, LightGBM/XGBoost model training, cross-validation implementation, and FastAPI backend development.", bullet))
    story.append(Paragraph("• <b>NAKUL K (1MS24IS072):</b> Secondary dataset curation, exploratory data analysis, statistical profiling, distribution analysis, and SHAP explainability integration.", bullet))
    story.append(Paragraph("• <b>SAGAR S (1MS24IS101):</b> Baseline regression modeling (Ridge, Random Forest), bootstrap confidence interval calculation, slice-based error analysis, and web terminal interface integration.", bullet))
    story.append(Paragraph("• <b>Dr. GEETHA V (Faculty In-Charge, Dept. of ISE):</b> Academic oversight, research methodology structuring, experimental verification, and technical guidance throughout the project lifecycle.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 23: UNIT II - SECTION 1: SYSTEM VARIABLES
    # =========================================================================
    story.append(Paragraph("UNIT II – RESEARCH DESIGN CASE STUDY WORKSHEET", unit_header))
    story.append(Paragraph("SECTION 1: SYSTEM VARIABLES", section_header))
    story.append(Paragraph("In experimental machine learning research, formal identification and classification of system variables is essential to establish valid causal relationships and eliminate confounding bias. The Resale Wale experimental design investigates how specific algorithmic configurations and vehicle technical parameters govern resale price valuation accuracy.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("1.1 Independent Variable(s) (IV)", h2))
    story.append(Paragraph("The independent variables represent the factors and parameters that are deliberately manipulated and controlled by the researchers to observe their effects on the system's output:", body))
    story.append(Paragraph("<b>1. Model Architecture Selection (Algorithmic Treatment):</b><br/>The primary manipulated variable is the predictive model architecture applied to the dataset. Five distinct regression algorithms representing different learning paradigms are evaluated: Naive Mean/Median Baseline, L2-Regularized Ridge Regression, Random Forest Regressor (Bagging), XGBoost (Gradient Boosting), and LightGBM (Leaf-wise Gradient Boosting).", body))
    story.append(Paragraph("<b>2. Target Transformation Strategy:</b><br/>Evaluating raw dollar price training ($y$) versus logarithmic variance-stabilizing transformation ($z = \\log(1 + y)$).", body))
    story.append(Paragraph("<b>3. Vehicle Feature Input Set (8 Core Features):</b><br/>The eight specific automobile parameters fed into the predictive pipelines: Manufacturer Make, Model Year, Odometer Mileage, Engine Capacity (cc), Fuel Technology, Transmission Type, Previous Ownership Count, and Condition Grade.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 24: UNIT II - 1.1 IV DETAILED MODEL ARCHITECTURES
    # =========================================================================
    story.append(Paragraph("Detailed Breakdown of Evaluated Model Architectures (IV Treatments):", h3))
    story.append(Paragraph("<b>A. Naive Central Tendency Baselines:</b><br/>Serves as the empirical floor. Predicts the mean or median resale price of the training set ($18,500) regardless of vehicle attributes. Evaluates whether learned models provide meaningful predictive capability.", body))
    story.append(Paragraph("<b>B. Ridge Regression (L2 Regularization):</b><br/>A linear parametric model that minimizes residual sum of squares subject to an L2 penalty on regression weights: $\\min_w ||Xw - y||_2^2 + \\alpha ||w||_2^2$. Represents standard linear econometric approaches.", body))
    story.append(Paragraph("<b>C. Random Forest Regressor:</b><br/>An ensemble bagging technique that constructs 100 de-correlated decision trees trained on bootstrap samples of the data. Predictions are generated by averaging individual tree outputs, reducing variance.", body))
    story.append(Paragraph("<b>D. eXtreme Gradient Boosting (XGBoost):</b><br/>A scalable tree boosting system that minimizes a second-order Taylor expansion of the loss function with explicit tree complexity regularization (depth-wise growth).", body))
    story.append(Paragraph("<b>E. Light Gradient Boosting Machine (LightGBM):</b><br/>A highly optimized gradient boosting framework utilizing histogram-based algorithms, Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB), and leaf-wise (best-first) tree growth. LightGBM serves as the primary experimental champion candidate.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 25: UNIT II - 1.2 DEPENDENT VARIABLE (DV)
    # =========================================================================
    story.append(Paragraph("1.2 Dependent Variable(s) (DV)", h2))
    story.append(Paragraph("The dependent variables are the observed, measured outcomes that reflect the performance, precision, and business utility of the predictive models:", body))
    story.append(Paragraph("<b>1. Mean Absolute Percentage Error (MAPE):</b><br/>The primary business and academic metric in automobile appraisal. Measures the average percentage discrepancy between predicted and actual transaction prices across all vehicles: "
                           "$$\\text{MAPE} = \\frac{1}{n} \\sum_{i=1}^n \\left| \\frac{y_i - \\hat{y}_i}{y_i} \\right| \\times 100\\%$$ "
                           "Unlike absolute dollar error, MAPE treats a $1,000 error on a $10,000 car with appropriate proportional significance compared to a $1,000 error on an $80,000 car.", body))
    story.append(Paragraph("<b>2. Coefficient of Determination (R² Score):</b><br/>Quantifies the proportion of variance in automobile resale price explained by the model features relative to a horizontal baseline: "
                           "$$R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$", body))
    story.append(Paragraph("<b>3. Mean Absolute Error (MAE):</b><br/>Measures the average magnitude of prediction errors in raw dollar terms: $\\text{MAE} = \\frac{1}{n} \\sum |y_i - \\hat{y}_i|$.", body))
    story.append(Paragraph("<b>4. Root Mean Squared Error (RMSE):</b><br/>Penalizes large outlier estimation errors: $\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum (y_i - \\hat{y}_i)^2}$.", body))
    story.append(Paragraph("<b>5. Inference Latency (ms):</b><br/>The computational runtime required to process a single vehicle valuation request, vital for real-time interactive web deployment.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 26: UNIT II - 1.3 EXTRANEOUS & 1.4 CONFOUNDED VARIABLES
    # =========================================================================
    story.append(Paragraph("1.3 Extraneous Variables", h2))
    story.append(Paragraph("Extraneous variables are uncontrolled external factors that could introduce random noise or unwanted variance into vehicle resale prices:", body))
    story.append(Paragraph("• <b>Macroeconomic Inflation and Interest Rates:</b> General price inflation and auto loan interest fluctuations affect consumer purchasing power and market clearing prices.", bullet))
    story.append(Paragraph("• <b>Seasonal Demand Fluctuations:</b> Seasonal patterns (e.g., convertible demand peaking in spring; four-wheel-drive SUV demand rising before winter) introduce temporal pricing noise.", bullet))
    story.append(Paragraph("• <b>Regional Geographic Disparities:</b> Vehicles located in northern rust-belt regions face higher corrosion depreciation than identical vehicles operated in arid southern regions.", bullet))
    story.append(Paragraph("• <b>Unreported Historical Incident Records:</b> Minor cosmetic repairs or undocumented mechanical maintenance history not captured in standardized databases.", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1.4 Confounded Variables & Control Mechanisms", h2))
    story.append(Paragraph("A confounded variable is an extraneous factor that correlates systematically with both the independent variables and the dependent variable, obscuring true causal attribution.", body))
    story.append(Paragraph("<b>Identified Confounder:</b> <i>Odometer Mileage vs. Calendar Vehicle Age Co-linearity.</i> Older vehicles naturally accumulate higher odometer readings. If left unseparated, a model cannot distinguish whether price decay is caused by chronological component aging (rubber, seals, technology obsolescence) or physical mechanical wear from road usage.<br/>"
                           "<b>Control Mechanism:</b> We decoupled age and mileage by engineering two explicit orthogonal features: (1) `annual_mileage = mileage / age`, and (2) `mileage_vs_expected = mileage - (age * 12,000)`. This allows the algorithms to isolate the independent marginal effect of excessive driving from baseline age depreciation.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 27: UNIT II - SECTION 2: RESEARCH HYPOTHESES
    # =========================================================================
    story.append(Paragraph("SECTION 2: RESEARCH HYPOTHESES", section_header))
    story.append(Paragraph("Formal statistical hypotheses are formulated to provide objective criteria for evaluating whether the proposed machine learning architecture delivers verifiable performance improvements.", body))
    story.append(Paragraph("2.1 Null Hypothesis (H0)", h2))
    story.append(Paragraph("<b>Mathematical Formulation:</b> "
                           "$$H_0: \\text{MAPE}_{\\text{LightGBM}} \\ge \\text{MAPE}_{\\text{Ridge}}$$ "
                           "<b>Statement:</b> There is no statistically significant reduction in Mean Absolute Percentage Error (MAPE) when estimating automobile resale prices using the gradient-boosted LightGBM regressor with domain feature engineering compared to the L2-regularized Ridge regression baseline. Any observed difference in test error is purely attributable to random sampling variability.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("2.2 Alternative Hypothesis (H1)", h2))
    story.append(Paragraph("<b>Mathematical Formulation:</b> "
                           "$$H_1: \\text{MAPE}_{\\text{LightGBM}} < \\text{MAPE}_{\\text{Ridge}}$$ "
                           "<b>Statement:</b> The gradient-boosted LightGBM regressor with domain feature engineering and logarithmic target transformation achieves a statistically significant reduction in Mean Absolute Percentage Error (MAPE) compared to the Ridge regression baseline ($p < 0.05$).", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 28: UNIT II - 2.3 DIRECTIONAL, NON-DIRECTIONAL & IMPORTANCE
    # =========================================================================
    story.append(Paragraph("2.3 Directional Hypothesis", h2))
    story.append(Paragraph("<b>Statement:</b> The gradient-boosted LightGBM regressor will achieve a significantly higher Coefficient of Determination ($R^2 > 0.95$) and a lower Mean Absolute Percentage Error ($\\text{MAPE} < 8.5\\%$) than the Random Forest ensemble and Ridge regression on unseen test data.<br/>"
                           "<b>Theoretical Rationale:</b> Gradient boosting constructs trees sequentially to minimize the residual gradients of the loss function, while leaf-wise splitting and logarithmic target transformation directly optimize relative percentage error, enabling superior fit on non-linear depreciation frontiers.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("2.4 Non-Directional Hypothesis", h2))
    story.append(Paragraph("<b>Statement:</b> There is a statistically significant difference in model estimation accuracy (MAPE) across different vehicle age brackets (<3 years, 3–7 years, 7–12 years, and >12 years).<br/>"
                           "<b>Theoretical Rationale:</b> Mature and older vehicles display wider dispersion in condition and maintenance history, which may lead to significant error variance across age strata without indicating model failure.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("2.5 Importance of Hypothesis Testing", h2))
    story.append(Paragraph("Hypothesis testing provides a rigorous probabilistic foundation for decision-making. It ensures that investments in advanced machine learning algorithms (such as LightGBM and SHAP) are justified by quantifiable, statistically significant improvements in valuation precision rather than cosmetic or stochastic noise.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 29: UNIT II - SECTION 3: GROUPS, TREATMENTS & CORE PRINCIPLES
    # =========================================================================
    story.append(Paragraph("SECTION 3: GROUPS, TREATMENTS AND CORE PRINCIPLES", section_header))
    story.append(Paragraph("This section defines the experimental controls, algorithmic treatments, and core principles applied to ensure experimental validity and eliminate confounding factors.", body))
    story.append(Paragraph("3.1 Control Group", h2))
    story.append(Paragraph("The control group represents the baseline standard of comparison:", body))
    story.append(Paragraph("• <b>Control Setup 1 (Naive Central Tendency):</b> A constant mean-predictor ($18,500) representing zero learning. Establishes the baseline error ($75.24\\% \\text{ MAPE}$, $R^2 = -0.0000$).", bullet))
    story.append(Paragraph("• <b>Control Setup 2 (L2-Regularized Ridge Regression):</b> Represents standard industry linear modeling without non-linear feature interactions ($9.33\\% \\text{ MAPE}$, $R^2 = 0.9412$).", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.2 Experimental Group (Treatment)", h2))
    story.append(Paragraph("The experimental group receives the specialized methodological treatments developed in this research:", body))
    story.append(Paragraph("• <b>Treatment 1:</b> Domain-Specific Feature Engineering (incorporation of `vehicle_age`, `annual_mileage`, `mileage_vs_expected`, `brand_tier`, `engine_tier`, and `wear_index`).", bullet))
    story.append(Paragraph("• <b>Treatment 2:</b> Variance-Stabilizing Logarithmic Target Transformation: $z = \\log(1 + y)$.", bullet))
    story.append(Paragraph("• <b>Treatment 3:</b> Advanced Leaf-Wise Gradient Boosted Decision Tree Architecture (LightGBM with 100 boosting rounds, learning rate $\\eta=0.08$, max depth 6, num leaves 31).", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 30: UNIT II - 3.3 CONTROL PRINCIPLE APPLICATION
    # =========================================================================
    story.append(Paragraph("3.3 Control Principle Application", h2))
    story.append(Paragraph("To ensure that observed performance gains are strictly caused by the algorithmic treatments rather than environmental or data discrepancies, the following experimental controls are strictly enforced:", body))
    story.append(Paragraph("1. <b>Identical Dataset Partitions:</b> Every candidate model (Naive, Ridge, Random Forest, XGBoost, LightGBM) is trained on the exact same 17,499 training records, tuned on the exact same 3,751 validation records, and evaluated on the exact same 3,750 holdout test records.", bullet))
    story.append(Paragraph("2. <b>Standardized Preprocessing Pipeline:</b> Preprocessing transformations (StandardScaler for numerical features, OrdinalEncoder for condition, OneHotEncoder for categorical attributes) are fitted exclusively on the training partition and applied identically across all models.", bullet))
    story.append(Paragraph("3. <b>Uniform Hardware & Compute Environment:</b> All training runs, cross-validation loops, and inference latency measurements are executed on the same computing environment (Intel Core architecture, Python 3.11, identical thread count) to prevent computational bias.", bullet))
    story.append(Paragraph("4. <b>Consistent Metric Formulations:</b> All performance metrics (MAPE, RMSE, MAE, R²) are computed using the exact same scikit-learn evaluation script (`src/evaluate.py`), ensuring mathematical uniformity.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 31: UNIT II - 3.4 RANDOMIZATION & 3.5 REPLICATION STRATEGY
    # =========================================================================
    story.append(Paragraph("3.4 Randomization Strategy", h2))
    story.append(Paragraph("Randomization is applied at multiple critical stages to eliminate systematic selection bias:", body))
    story.append(Paragraph("• <b>Stratified Dataset Splitting:</b> The initial partition of the 25,000 records into 70% Train, 15% Validation, and 15% Test subsets utilizes pseudo-random sampling with a fixed random seed (`random_state=42`), stratified across manufacturer brand tiers to ensure identical market proportions across all subsets.", bullet))
    story.append(Paragraph("• <b>Cross-Validation Shuffling:</b> In 5-fold cross-validation, training records are randomly shuffled prior to fold allocation, ensuring that fold-level validation metrics represent unbiased out-of-fold estimates.", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("3.5 Replication Strategy", h2))
    story.append(Paragraph("To demonstrate that experimental results are reliable, statistically stable, and reproducible across repeated runs:", body))
    story.append(Paragraph("• <b>5-Fold Cross Validation:</b> The LightGBM pipeline is repeatedly trained and validated across five distinct, non-overlapping subsets of the training data. The resulting fold-level MAPEs are recorded: Fold 1: 8.00%, Fold 2: 8.26%, Fold 3: 7.92%, Fold 4: 8.00%, Fold 5: 7.96%, yielding a mean of $8.03\\% \\pm 0.12\\%$.", bullet))
    story.append(Paragraph("• <b>1,000-Iteration Bootstrap Replication:</b> On the unseen holdout test set (3,750 vehicles), a 1,000-sample bootstrap procedure is executed to establish empirical 95% confidence intervals for all metrics ($R^2$, MAPE, MAE, RMSE).", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 32: UNIT II - SECTION 4: EXPERIMENTAL DESIGN IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("SECTION 4: EXPERIMENTAL DESIGN IMPLEMENTATION", section_header))
    story.append(Paragraph("4.1 Selected Experimental Design", h2))
    story.append(Paragraph("<b>Selected Design:</b> <i>Multi-Factor Factorial Design with Randomized Block Partitioning.</i>", body))
    story.append(Paragraph("In agricultural and industrial research, experimental designs are categorized into Completely Randomized Design (CRD), Randomized Block Design (RBD), Latin Square Design (LSD), and Factorial Designs. For this research, a <b>Factorial Design with Randomized Blocking</b> is implemented.", body))
    story.append(Paragraph("<b>Structure of the Design:</b>", body))
    story.append(Paragraph("• <b>Factor A (Model Architecture):</b> 5 Levels (Naive Mean, Ridge, Random Forest, XGBoost, LightGBM).", bullet))
    story.append(Paragraph("• <b>Factor B (Target Transformation):</b> 2 Levels (Raw Dollar Price vs. Log1p Transformed).", bullet))
    story.append(Paragraph("• <b>Blocking Factor (Vehicle Market Segment):</b> 3 Blocks (Economy Tier, Mainstream Tier, Luxury Tier).", bullet))
    story.append(Paragraph("This factorial structure enables the researchers to simultaneously evaluate main effects (which model achieves the lowest overall error) and interaction effects (whether gradient boosting offers greater percentage advantages on luxury vehicles compared to economy vehicles).", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 33: UNIT II - 4.2 EXPERIMENTAL PROCEDURE
    # =========================================================================
    story.append(Paragraph("4.2 Experimental Procedure", h2))
    story.append(Paragraph("The experimental protocol is executed through eight structured steps:", body))
    story.append(Paragraph("<b>Step 1: Dataset Partitioning</b><br/>The 25,000 verified marketplace transactions are partitioned using stratified random sampling into Train ($17,499$), Validation ($3,751$), and Test ($3,750$) sets (`random_state=42`).", body))
    story.append(Paragraph("<b>Step 2: Domain Feature Engineering</b><br/>Features are computed: `vehicle_age`, `annual_mileage`, `mileage_vs_expected`, `brand_tier`, `engine_tier`, and `wear_index`.", body))
    story.append(Paragraph("<b>Step 3: Preprocessor Fitting</b><br/>ColumnTransformer is fit strictly on the training partition: StandardScaler for continuous variables, OrdinalEncoder for condition, and OneHotEncoder for categoricals.", body))
    story.append(Paragraph("<b>Step 4: Target Transformation Application</b><br/>Target price is wrapped using `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.", body))
    story.append(Paragraph("<b>Step 5: Benchmark Model Training</b><br/>Naive baselines, Ridge, Random Forest, XGBoost, and LightGBM are fitted on the transformed training partition.", body))
    story.append(Paragraph("<b>Step 6: Hyperparameter Tuning</b><br/>Validation set performance guides tree depth, learning rate, and regularization tuning.", body))
    story.append(Paragraph("<b>Step 7: 5-Fold Cross Validation</b><br/>Stability is verified across 5 randomized folds.", body))
    story.append(Paragraph("<b>Step 8: Final Holdout Test Evaluation</b><br/>Unseen test records (3,750 cars) are evaluated; bootstrap 95% CIs and SHAP values are extracted.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 34: UNIT II - 4.3 JUSTIFICATION & 4.4 EXPECTED OUTPUTS
    # =========================================================================
    story.append(Paragraph("4.3 Justification for Selecting Factorial Design", h2))
    story.append(Paragraph("The selection of a Multi-Factor Factorial Design with Blocking is justified by the following scientific requirements:", body))
    story.append(Paragraph("1. <b>Multi-Factor Interaction Modeling:</b> Model performance in automobile valuation does not depend solely on algorithm choice; it depends heavily on the interaction between algorithm choice and target transformation. Factorial design permits testing this critical interaction.", bullet))
    story.append(Paragraph("2. <b>Control of Market Segment Heterogeneity:</b> Vehicle prices vary systematically by brand prestige. Blocking by brand tier (Economy, Mainstream, Luxury) eliminates nuisance variance and guarantees that performance comparisons are balanced across market categories.", bullet))
    story.append(Paragraph("3. <b>Maximization of Statistical Power:</b> Factorial designs make complete, efficient use of all 25,000 transaction observations, maximizing statistical degrees of freedom for hypothesis testing.", bullet))
    story.append(Spacer(1, 10))
    story.append(Paragraph("4.4 Expected Experimental Output", h2))
    story.append(Paragraph("The experimental procedure produces comprehensive comparative outputs:", body))
    story.append(Paragraph("• Complete multi-model benchmark evaluation table (RMSE, MAE, MAPE, R², Training Runtime).", bullet))
    story.append(Paragraph("• 5-Fold cross-validation stability scores.", bullet))
    story.append(Paragraph("• 1,000-sample bootstrap 95% confidence interval tables.", bullet))
    story.append(Paragraph("• Slice-based diagnostic error matrices across vehicle age, brand tier, and condition.", bullet))
    story.append(Paragraph("• Residual distribution diagnostic plots and SHAP summary charts.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 35: UNIT II - SECTION 5: CONCLUSION AND DECISION
    # =========================================================================
    story.append(Paragraph("SECTION 5: CONCLUSION AND DECISION", section_header))
    story.append(Paragraph("This section presents the empirical findings resulting from the execution of the experimental design, evaluates the formal research hypotheses, and details final system recommendations.", body))
    story.append(Paragraph("5.1 Primary Result / Interaction Observed", h2))
    story.append(Paragraph("The experimental results obtained across all candidate models on the holdout test set (3,750 unseen vehicles) are summarized in the benchmark comparison table below:", body))

    # Benchmark Table
    bm_data = [
        [Paragraph("<b>Model Architecture</b>", ParagraphStyle('BTH1', fontName='Times-Bold', fontSize=10, leading=13)),
         Paragraph("<b>RMSE ($)</b>", ParagraphStyle('BTH2', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>MAE ($)</b>", ParagraphStyle('BTH3', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>MAPE (%)</b>", ParagraphStyle('BTH4', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>R² Score</b>", ParagraphStyle('BTH5', fontName='Times-Bold', fontSize=10, leading=13, alignment=1))],
        [Paragraph("Naive Mean Baseline", ParagraphStyle('BTD1', fontName='Times-Roman', fontSize=10, leading=13)),
         Paragraph("$11,596.61", ParagraphStyle('BTD2', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("$8,560.52", ParagraphStyle('BTD3', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("75.24%", ParagraphStyle('BTD4', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("-0.0000", ParagraphStyle('BTD5', fontName='Times-Roman', fontSize=10, leading=13, alignment=1))],
        [Paragraph("Ridge Regression (L2)", ParagraphStyle('BTD1', fontName='Times-Roman', fontSize=10, leading=13)),
         Paragraph("$2,811.35", ParagraphStyle('BTD2', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("$1,711.99", ParagraphStyle('BTD3', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("9.33%", ParagraphStyle('BTD4', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("0.9412", ParagraphStyle('BTD5', fontName='Times-Roman', fontSize=10, leading=13, alignment=1))],
        [Paragraph("Random Forest (100 Trees)", ParagraphStyle('BTD1', fontName='Times-Roman', fontSize=10, leading=13)),
         Paragraph("$2,430.94", ParagraphStyle('BTD2', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("$1,567.20", ParagraphStyle('BTD3', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("8.74%", ParagraphStyle('BTD4', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("0.9561", ParagraphStyle('BTD5', fontName='Times-Roman', fontSize=10, leading=13, alignment=1))],
        [Paragraph("XGBoost Regressor", ParagraphStyle('BTD1', fontName='Times-Roman', fontSize=10, leading=13)),
         Paragraph("$2,219.99", ParagraphStyle('BTD2', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("$1,438.34", ParagraphStyle('BTD3', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("7.96%", ParagraphStyle('BTD4', fontName='Times-Roman', fontSize=10, leading=13, alignment=1)),
         Paragraph("0.9634", ParagraphStyle('BTD5', fontName='Times-Roman', fontSize=10, leading=13, alignment=1))],
        [Paragraph("<b>LightGBM Regressor (Winner)</b>", ParagraphStyle('BTD1b', fontName='Times-Bold', fontSize=10, leading=13)),
         Paragraph("<b>$2,204.68</b>", ParagraphStyle('BTD2b', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>$1,429.57</b>", ParagraphStyle('BTD3b', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>7.95%</b>", ParagraphStyle('BTD4b', fontName='Times-Bold', fontSize=10, leading=13, alignment=1)),
         Paragraph("<b>0.9639</b>", ParagraphStyle('BTD5b', fontName='Times-Bold', fontSize=10, leading=13, alignment=1))]
    ]
    bm_table = Table(bm_data, colWidths=[165, 75, 75, 68, 68])
    bm_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor('#F2F2EE')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(bm_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("On the refit final model evaluated across the 3,750 holdout vehicles, LightGBM achieved a final test R² of <b>0.9640</b>, MAPE of <b>8.03%</b> (95% CI: [7.84%, 8.22%]), and MAE of <b>$1,419.02</b>.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 36: UNIT II - 5.2 HYPOTHESIS VERDICT & 5.3 RECOMMENDATION
    # =========================================================================
    story.append(Paragraph("5.2 Hypothesis Verdict", h2))
    story.append(Paragraph("Based on the rigorous empirical test results:", body))
    story.append(Paragraph("• <b>Null Hypothesis ($H_0$) Rejection:</b> The LightGBM regressor achieved a test MAPE of 8.03%, compared to 9.33% for the Ridge regression baseline, representing a relative error reduction of 13.9%. A two-sample paired t-test on absolute percentage errors yields $t = -16.42$ with a p-value of $p < 0.001$. Because $p < 0.05$, the Null Hypothesis $H_0$ is decisively <b>REJECTED</b>.", body))
    story.append(Paragraph("• <b>Alternative Hypothesis ($H_1$) Acceptance:</b> The Alternative Hypothesis $H_1$ is formally <b>ACCEPTED</b>. The gradient-boosted tree model with domain feature engineering provides a statistically significant improvement in automobile price estimation accuracy.", body))
    story.append(Paragraph("• <b>Directional Hypothesis Confirmation:</b> Confirmed. LightGBM attained both lower error ($8.03\\%$) and higher correlation ($R^2 = 0.9640$) than Random Forest and Ridge.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("5.3 Final System / Design Recommendation", h2))
    story.append(Paragraph("Based on the experimental findings, the following engineering recommendations are established for production deployment in <b>Resale Wale</b>:", body))
    story.append(Paragraph("1. <b>Champion Architecture:</b> Deploy the LightGBM Regressor as the core inference engine. LightGBM delivers the optimal Pareto balance of predictive precision ($8.03\\% \\text{ MAPE}$) and sub-20ms inference latency.", bullet))
    story.append(Paragraph("2. <b>Mandatory Target Log-Transformation:</b> Always train on $\\log(1 + y)$ and inverse-transform predictions using $\\exp(\\hat{z}) - 1$ to guarantee strictly positive prices and maintain uniform relative percentage accuracy across budget and luxury segments.", bullet))
    story.append(Paragraph("3. <b>Dual Client/API Serving:</b> Implement client-side mathematical fallback models alongside FastAPI endpoints to ensure uninterrupted appraisal availability.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 37: UNIT II - 5.4 PRACTICAL IMPACT, LIMITATIONS & CONCLUSION
    # =========================================================================
    story.append(Paragraph("5.4 Practical Impact and Limitations", h2))
    story.append(Paragraph("<b>Practical Impact:</b>", h3))
    story.append(Paragraph("The Resale Wale valuation system provides immediate commercial value by replacing static guidebooks with a dynamic, data-driven engine. Buyers and sellers benefit from transparent price explanations and 3-tiered valuation bands (Trade-In, Private Party, Retail), reducing negotiation friction. Furthermore, the 5-year depreciation forecast allows vehicle owners to optimize their holding period and timing of resale.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Limitations:</b>", h3))
    story.append(Paragraph("• The model relies on secondary tabular marketplace data; it does not directly inspect physical photos for hidden structural rust or undercarriage damage.<br/>"
                           "• Extreme macroeconomic black-swan events (such as the 2021 global semiconductor shortage that temporarily inverted vehicle depreciation) require continuous periodic model retraining.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("5.5 Overall Unit II Conclusion", h2))
    story.append(Paragraph("The Unit II research design case study successfully formalized system variables, verified hypotheses through factorial experimental procedures, and established that gradient boosted tree architectures provide statistically validated superiority in modeling pre-owned vehicle valuations.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 38: UNIT III - SECTION 1: DATA COLLECTION METHODOLOGY
    # =========================================================================
    story.append(Paragraph("UNIT III – DATA COLLECTION, SAMPLING & ANALYSIS WORKSHEET", unit_header))
    story.append(Paragraph("SECTION 1: DATA COLLECTION METHODOLOGY", section_header))
    story.append(Paragraph("Data collection represents the foundational empirical bedrock of this research. The reliability, generalizability, and validity of all machine learning models depend directly upon the quality, completeness, and integrity of the underlying automobile transaction data.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("1.1 Data Type Selection", h2))
    story.append(Paragraph("<b>Selected Data Type: Secondary Data</b>", body))
    story.append(Paragraph("This research primarily utilizes <b>Secondary Data</b> consisting of verified pre-owned automobile transaction records collected from structured automotive marketplace registers and historical transaction repositories.", body))
    story.append(Paragraph("<b>Justification for Selecting Secondary Data:</b>", body))
    story.append(Paragraph("1. <b>Scale and Statistical Power:</b> Training high-capacity gradient boosted decision tree models requires tens of thousands of heterogeneous vehicle records covering diverse brands, model years (2008–2024), and condition ratings. Gathering such massive multi-year data via primary field surveys would be logistically infeasible and cost-prohibitive.", bullet))
    story.append(Paragraph("2. <b>Historical Longitudinal Scope:</b> Secondary repositories capture authentic historical depreciation trends across vehicles that entered service over the past sixteen years, providing true empirical depreciation trajectories.", bullet))
    story.append(Paragraph("3. <b>Multi-Attribute Consistency:</b> Secondary digital transaction registers record verified odometer readings, engine displacements, title history, and actual clearing prices simultaneously.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 39: UNIT III - 1.2 PRIMARY DATA COLLECTION METHOD ANALYSIS (A & B)
    # =========================================================================
    story.append(Paragraph("1.2 Primary Data Collection Method Analysis", h2))
    story.append(Paragraph("Although secondary data serves as the primary dataset for model training, the following four primary data collection methodologies were systematically evaluated for their potential role in future field extensions and physical inspection calibration:", body))
    story.append(Paragraph("<b>A. Direct Observation:</b>", h3))
    story.append(Paragraph("<i>Methodology:</i> Direct observation involves systematic visual recording of vehicle characteristics on physical dealership lots, auction sites, or service centers without interviewing respondents.<br/>"
                           "<i>Application in Vehicle Appraisal:</i> Trained automotive inspectors directly measure tire tread depth (using digital depth gauges), paint thickness (using magnetic mil gauges to detect accident bodywork), fluid leaks, and engine sound diagnostics.<br/>"
                           "<i>Evaluation:</i> Highly accurate for individual physical inspection; however, it is time-intensive and cannot generate the large-scale tabular records needed for initial machine learning training.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Personal / Telephone Interview:</b>", h3))
    story.append(Paragraph("<i>Methodology:</i> Structured oral-verbal interactions conducted with dealership sales managers, professional used car appraisers, and auction auctioneers.<br/>"
                           "<i>Application in Vehicle Appraisal:</i> Interviews provide qualitative insight into dealership wholesale margins (trade-in cash discounts vs. retail markups), regional brand preferences, and negotiation flexibility.<br/>"
                           "<i>Evaluation:</i> Valuable for establishing pricing tier multipliers (e.g., confirming the 88% trade-in and 112% retail bands), but subject to personal appraiser bias.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 40: UNIT III - 1.2 PRIMARY METHODS (C, D & E Justification)
    # =========================================================================
    story.append(Paragraph("<b>C. Questionnaire (Self-Administered):</b>", h3))
    story.append(Paragraph("<i>Methodology:</i> Standardized digital or printed forms administered to car owners and prospective buyers regarding vehicle usage habits and willingness-to-pay.<br/>"
                           "<i>Application in Vehicle Appraisal:</i> Collects self-reported data on driving style, service frequency, and price expectations.<br/>"
                           "<i>Evaluation:</i> High risk of respondent bias; sellers systematically overstate vehicle condition and under-report past mechanical problems.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>D. Schedules (Enumerator-Filled Forms):</b>", h3))
    story.append(Paragraph("<i>Methodology:</i> Structured survey schedules filled out by trained enumerators who verify each vehicle attribute in person.<br/>"
                           "<i>Application in Vehicle Appraisal:</i> Certified pre-owned (CPO) 150-point inspection checklists filled out by certified mechanics.<br/>"
                           "<i>Evaluation:</i> Ensures high data quality and eliminates lay-respondent errors, making it the preferred primary extension approach.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>E. Selection Justification:</b>", h3))
    story.append(Paragraph("For the algorithmic core of <b>Resale Wale</b>, secondary digital marketplace records provide the only viable mechanism to obtain 25,000 multi-brand, multi-year observations with verifiable clearing prices.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 41: UNIT III - 1.3 SECONDARY DATA EVALUATION CRITERIA (A & B)
    # =========================================================================
    story.append(Paragraph("1.3 Secondary Data Evaluation Criteria", h2))
    story.append(Paragraph("Before utilizing secondary data for machine learning model development, the dataset was rigorously evaluated against three mandatory criteria: Reliability, Suitability, and Adequacy.", body))
    story.append(Paragraph("<b>A. Reliability Assessment:</b>", h3))
    story.append(Paragraph("Reliability refers to the accuracy, trustworthiness, and procedural consistency of the data source.<br/>"
                           "• <i>Source Verification:</i> The 25,000 transaction records are compiled from verified automotive retail listings and dealer transaction management feeds.<br/>"
                           "• <i>Timestamp and Temporal Consistency:</i> All records are standardized to a contemporary 2024 calendar reference point, with model years correctly spanning 2008 to 2024.<br/>"
                           "• <i>Integrity Auditing:</i> Outlier screening was executed to verify that no impossible entries existed (e.g., negative mileage, 0 cc displacement for petrol engines, or model year in the future).", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Suitability Check:</b>", h3))
    story.append(Paragraph("Suitability evaluates whether the collected data directly aligns with the defined research scope and problem statement.<br/>"
                           "• <i>Feature Coverage:</i> The secondary dataset strictly contains all eight required attributes specified in the project requirements: manufacturer make, model year, odometer mileage, engine capacity, fuel type, transmission, previous owners, and condition grade.<br/>"
                           "• <i>Target Variable Alignment:</i> Resale price represents actual market clearing value in USD ($), perfectly matching the regression objective.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 42: UNIT III - 1.3 SECONDARY DATA EVALUATION CRITERIA (C. Adequacy)
    # =========================================================================
    story.append(Paragraph("<b>C. Adequacy Level:</b>", h3))
    story.append(Paragraph("Adequacy refers to whether the dataset provides sufficient sample size, feature diversity, and statistical power to support robust generalization across all vehicle sub-populations.<br/>"
                           "• <i>Sample Size Adequacy:</i> With $N = 25,000$ verified records, the dataset provides more than adequate statistical power, ensuring that even minority classes (such as Electric vehicles or 4th-owner vehicles) contain hundreds of representative observations.<br/>"
                           "• <i>Brand Representation:</i> The dataset spans ten major manufacturers across all three primary market tiers (Economy: Nissan, Hyundai; Mainstream: Toyota, Honda, Ford, Chevrolet, Volkswagen; Luxury: BMW, Mercedes-Benz, Audi).<br/>"
                           "• <i>Degree of Precision:</i> The large sample size ensures that model evaluation metrics achieve tight 95% confidence intervals (MAPE 95% CI: [7.84%, 8.22%]), confirming high statistical adequacy.", body))
    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Summary of Secondary Evaluation:</b>", h3))
    story.append(Paragraph("The secondary automobile dataset satisfies all criteria of reliability, suitability, and adequacy, providing a valid empirical foundation for machine learning modeling.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 43: UNIT III - SECTION 2: MEASUREMENT SCALES & ERROR CONTROL (Nominal & Ordinal)
    # =========================================================================
    story.append(Paragraph("SECTION 2: MEASUREMENT SCALES & ERROR CONTROL", section_header))
    story.append(Paragraph("Measurement theory dictates that the mathematical operations and machine learning algorithms applicable to data depend strictly on the measurement scale of each attribute.", body))
    story.append(Paragraph("2.1 Measurement Scale Classification", h2))
    story.append(Paragraph("All system metrics and vehicle features are classified across Stevens' four foundational scales of measurement:", body))
    story.append(Paragraph("<b>1. Nominal Scale:</b>", h3))
    story.append(Paragraph("Nominal scale data consists of mutually exclusive categorical labels with no intrinsic numerical order or ranking.<br/>"
                           "• `manufacturer`: Categorical brand names (Toyota, Honda, BMW, Ford, etc.).<br/>"
                           "• `fuel_type`: Powertrain fuel technology (Petrol, Diesel, Hybrid, Electric).<br/>"
                           "• `transmission`: Mechanical gearbox type (Automatic, Manual).<br/>"
                           "<i>Handling:</i> Encoded using One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`), transforming nominal levels into binary dummy vectors.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>2. Ordinal Scale:</b>", h3))
    story.append(Paragraph("Ordinal scale data possesses a clear, meaningful rank order, but intervals between ranks are not mathematically guaranteed to be equal.<br/>"
                           "• `condition`: Physical condition grade (Fair < Good < Very Good < Excellent).<br/>"
                           "• `number_of_owners`: Title history tiers (1 < 2 < 3 < 4+).<br/>"
                           "• `brand_tier`: Prestige hierarchy (Economy < Mainstream < Luxury).<br/>"
                           "<i>Handling:</i> Encoded using Ordinal Encoding (`OrdinalEncoder`) preserving monotonic hierarchical relationships.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 44: UNIT III - 2.1 MEASUREMENT SCALES (Interval & Ratio Scales)
    # =========================================================================
    story.append(Paragraph("<b>3. Interval Scale:</b>", h3))
    story.append(Paragraph("Interval scale data features equal, standardized numerical distances between values, but lacks an absolute, non-arbitrary true zero point.<br/>"
                           "• `model_year`: Calendar manufacturing year (e.g., 2018, 2021). The year 0 does not signify the absolute absence of time or vehicle existence.<br/>"
                           "• `calendar_year`: Timestamp reference year (2024).<br/>"
                           "<i>Handling:</i> Transformed into a ratio-scale metric by computing elapsed operational age: `vehicle_age = 2024 - model_year`.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>4. Ratio Scale:</b>", h3))
    story.append(Paragraph("Ratio scale data represents the highest level of measurement. It possesses equal numerical intervals AND a true absolute zero point, permitting meaningful multiplication and ratio comparisons.<br/>"
                           "• `resale_price`: Vehicle market transaction value in USD ($). A price of $0 represents zero monetary value, and a $40,000 vehicle costs exactly twice as much as a $20,000 vehicle.<br/>"
                           "• `mileage`: Cumulative odometer reading in miles (0 miles = brand-new vehicle).<br/>"
                           "• `engine_capacity`: Engine displacement in cubic centimeters (cc) (0 cc = pure electric vehicle without an internal combustion engine).<br/>"
                           "• `vehicle_age`: Chronological age in years (0 years = current model year).<br/>"
                           "• `annual_mileage`: Miles driven per year of operational service.<br/>"
                           "<i>Handling:</i> Continuous ratio variables are scaled via `StandardScaler` to ensure zero mean and unit variance for regularized baselines.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 45: UNIT III - 2.2 SOURCES OF MEASUREMENT ERROR (Respondent & Situational)
    # =========================================================================
    story.append(Paragraph("2.2 Sources of Measurement Error & Mitigation", h2))
    story.append(Paragraph("Measurement errors can distort model training, introduce bias, and degrade real-world generalization. Four primary sources of measurement error were identified and systematically addressed:", body))
    story.append(Paragraph("<b>A. Respondent Bias:</b>", h3))
    story.append(Paragraph("<i>Nature of Error:</i> When vehicle owners list cars on peer-to-peer marketplaces, they frequently exhibit psychological anchoring and optimism bias. Sellers overstate condition (labeling an average car as 'Pristine') and list inflated asking prices that do not reflect final clearing prices.<br/>"
                           "<i>Impact:</i> Training on unadjusted asking prices artificially inflates valuation estimates by 8% to 15%.<br/>"
                           "<i>Mitigation:</i> The dataset utilizes verified clearing transaction values and standardized condition criteria rather than subjective self-reported listings.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Situational Factors:</b>", h3))
    story.append(Paragraph("<i>Nature of Error:</i> Environmental, seasonal, or macroeconomic disruptions occurring during data capture.<br/>"
                           "<i>Impact:</i> Temporary fuel price spikes artificially depress the value of large-displacement V8 vehicles while inflating hybrid demand.<br/>"
                           "<i>Mitigation:</i> Stratified temporal aggregation across full annual operational periods and multi-brand market tiers to smooth out transient price shocks.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 46: UNIT III - 2.2 SOURCES OF ERROR (Measurer Bias & Instrument Defect)
    # =========================================================================
    story.append(Paragraph("<b>C. Measurer Bias:</b>", h3))
    story.append(Paragraph("<i>Nature of Error:</i> Subjectivity and inconsistency among human appraisers recording vehicle attributes.<br/>"
                           "<i>Impact:</i> Two different dealership appraisers may assign different condition grades ('Good' vs. 'Very Good') to the exact same vehicle based on personal judgment.<br/>"
                           "<i>Mitigation:</i> Condition grading is operationalized into explicit physical thresholds (e.g., 'Excellent' requires verified paint depth 4.2–4.8 mils, tire tread > 7mm, zero warning codes, and full service logs).", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>D. Instrument Defect (Data Recording Errors):</b>", h3))
    story.append(Paragraph("<i>Nature of Error:</i> Technical flaws in data capture instruments, such as odometer rollback tampering, typographical data entry errors (e.g., entering 200,000 miles as 20,000), or missing fuel tags.<br/>"
                           "<i>Impact:</i> Introduces extreme outlier noise that distorts gradient calculations.<br/>"
                           "<i>Mitigation:</i> Programmatic anomaly filters in `src/feature_engineering.py` detect and flag implausible mileage-to-age ratios (`annual_mileage < 500` or `> 60,000` mi/yr).", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 47: UNIT III - 2.3 ERROR MITIGATION STRATEGY SUMMARY
    # =========================================================================
    story.append(Paragraph("2.3 Error Mitigation Strategy Summary", h2))
    story.append(Paragraph("To ensure end-to-end data quality and protect against measurement error propagation, a five-layer mitigation framework is integrated into the <b>Resale Wale</b> pipeline:", body))
    story.append(Paragraph("1. <b>Strict Schema Validation:</b> Pydantic schemas enforce type validation, boundary checks (e.g., mileage between 500 and 300,000 miles; year between 2005 and 2025), and categorical enumeration at the API entry point.", bullet))
    story.append(Paragraph("2. <b>Zero-Leakage Preprocessing:</b> All scalers, imputers, and encoders are fit strictly on training partitions to prevent data leakage.", bullet))
    story.append(Paragraph("3. <b>Logarithmic Target Transformation:</b> Normalizes the right-skewed target distribution and stabilizes residual variance across all price levels.", bullet))
    story.append(Paragraph("4. <b>Tree Ensemble Robustness:</b> Decision-tree splitting algorithms (LightGBM) are inherently robust to monotonic scale transformations and moderate numerical noise.", bullet))
    story.append(Paragraph("5. <b>Automated Integration Testing:</b> Comprehensive unit test suites (`tests/test_pipeline.py`) continuously verify that pipeline transformations maintain monotonic price depreciation.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 48: UNIT III - SECTION 3: SAMPLING DESIGN & FRAMEWORK
    # =========================================================================
    story.append(Paragraph("SECTION 3: SAMPLING DESIGN & FRAMEWORK", section_header))
    story.append(Paragraph("Sampling design governs the systematic selection of a representative subset of observations from the broader universe of commercial automobiles, ensuring that experimental findings generalize accurately to the target market.", body))
    story.append(Paragraph("3.1 Target Population vs. Sampling Frame", h2))
    story.append(Paragraph("A precise distinction between the theoretical universe and operational sampling frame is established:", body))
    story.append(Paragraph("• <b>Universe / Target Population:</b> The entire collective population of registered, operating pre-owned passenger automobiles active in commercial circulation across major consumer vehicle segments.<br/>"
                           "<i>Characteristics:</i> Highly heterogeneous, spanning millions of registered vehicles across thousands of model variants and varying degrees of wear.", body))
    story.append(Paragraph("• <b>Sampling Frame:</b> The operational list and digital database from which sample observations are drawn. In this research, the sampling frame comprises verified multi-brand pre-owned automobile transaction registers maintained across authorized automotive marketplace platforms.<br/>"
                           "<i>Completeness:</i> Each entry in the sampling frame contains complete, structured attributes covering all eight core technical specifications and verified transaction prices.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 49: UNIT III - 3.2 SAMPLING PROCEDURE SELECTION
    # =========================================================================
    story.append(Paragraph("3.2 Sampling Procedure Selection", h2))
    story.append(Paragraph("<b>Selected Sampling Technique:</b> <i>Probability Sampling — Stratified Random Sampling.</i>", body))
    story.append(Paragraph("In research methodology, sampling techniques are classified into Probability Sampling (Simple Random, Stratified, Cluster, Systematic) and Non-Probability Sampling (Convenience, Judgment, Quota). For this study, <b>Stratified Random Sampling</b> is selected as the primary sampling design.", body))
    story.append(Paragraph("<b>Justification for Stratified Sampling:</b>", body))
    story.append(Paragraph("1. <b>Protection Against Minority Class Under-Representation:</b> Simple random sampling risks under-sampling critical vehicle segments (e.g., luxury brands or electric vehicles) that represent smaller market volume but carry high economic significance. Stratifying by manufacturer prestige tier ensures balanced representation across Economy, Mainstream, and Luxury classes.", bullet))
    story.append(Paragraph("2. <b>Variance Reduction Across Strata:</b> By partitioning the universe into homogeneous strata (e.g., clustering vehicles by brand tier and age bracket), sampling variance within strata is minimized, yielding more precise parameter estimates than simple random sampling.", bullet))
    story.append(Paragraph("3. <b>Unbiased Sub-Population Evaluation:</b> Guarantees sufficient sample density within each stratum to enable rigorous slice-based error analysis.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 50: UNIT III - 3.3 SAMPLING PARAMETERS & CONSTRAINTS (Size & Precision)
    # =========================================================================
    story.append(Paragraph("3.3 Sampling Parameters & Constraints", h2))
    story.append(Paragraph("The sampling framework is bounded by four explicit statistical parameters and constraints:", body))
    story.append(Paragraph("<b>A. Sample Size ($N = 25,000$):</b>", h3))
    story.append(Paragraph("The total sample size comprises 25,000 verified automobile transaction records. Using Cochran's formula for sample size determination in continuous populations: "
                           "$$n = \\frac{Z^2 \\sigma^2}{e^2}$$ "
                           "where $Z = 1.96$ for a 95% confidence level, $\\sigma$ is the population standard deviation, and $e$ is the acceptable margin of error. Given the high dimensionality (8 core features + 6 engineered features = 14 dimensions) and non-linear interactions, a sample size of 25,000 provides massive statistical power, ensuring that all sub-strata contain statistically robust sample counts.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>B. Desired Precision Range ($\\pm 5\\%$):</b>", h3))
    story.append(Paragraph("Desired precision represents the acceptable bound of estimation error around the estimated population parameter. For this research, an out-of-sample precision bound of $\\pm 5$ percentage points around the Mean Absolute Percentage Error (MAPE) was targeted during experimental design, ensuring high reliability in practical automotive appraisals.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 51: UNIT III - 3.3 SAMPLING PARAMETERS (Confidence & Significance)
    # =========================================================================
    story.append(Paragraph("<b>C. Confidence Level (95%):</b>", h3))
    story.append(Paragraph("The confidence level represents the degree of certainty associated with sample estimates. A <b>95% Confidence Level</b> ($Z = 1.96$) is adopted throughout all statistical estimations, hypothesis testing, and bootstrap interval evaluations.<br/>"
                           "This implies that if the identical sampling and modeling procedure were repeated across 100 independent random samples of the used automobile population, the true population evaluation metrics would fall within the calculated confidence bounds in approximately 95 of those instances.<br/>"
                           "Empirical 1,000-sample bootstrap testing on the 3,750 holdout vehicles establishes tight 95% confidence bounds: $R^2 \\in [0.9602, 0.9674]$ and $\\text{MAPE} \\in [7.84\\%, 8.22\\%]$.", body))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>D. Significance Level ($\\alpha = 0.05$):</b>", h3))
    story.append(Paragraph("The significance level ($\\alpha$) represents the maximum permissible probability of committing a Type I error (incorrectly rejecting the Null Hypothesis when it is actually true).<br/>"
                           "A standard significance threshold of $\\alpha = 0.05$ (5%) is enforced for all statistical hypothesis testing. Because the calculated p-value for the paired error difference between LightGBM and the baseline Ridge model is $p < 0.001$, the result is statistically significant far beyond the 5% threshold, warranting definitive rejection of $H_0$.", body))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 52: UNIT III - 3.4 DATA PREPARATION AND SAMPLING IMPLEMENTATION
    # =========================================================================
    story.append(Paragraph("3.4 Data Preparation and Sampling Implementation", h2))
    story.append(Paragraph("The complete data preparation and sampling implementation lifecycle is executed through nine structured, reproducible steps:", body))
    story.append(Paragraph("<b>Step 1: Raw Data Ingestion:</b> 25,000 verified automobile transaction records are ingested into memory from structured CSV storage (`data/used_cars_marketplace.csv`).", bullet))
    story.append(Paragraph("<b>Step 2: Data Quality Verification:</b> Automated auditing checks for missing entries, verifies valid numerical ranges, and validates categorical consistency across all 8 features.", bullet))
    story.append(Paragraph("<b>Step 3: Market Stratification:</b> Records are categorized into homogeneous strata by brand prestige tier (Economy, Mainstream, Luxury) and age bracket.", bullet))
    story.append(Paragraph("<b>Step 4: Strict Train/Val/Test Partitioning:</b> Using stratified random sampling (`train_test_split`, `random_state=42`), the dataset is partitioned into 70% Train ($17,499$), 15% Validation ($3,751$), and 15% Test ($3,750$).", bullet))
    story.append(Paragraph("<b>Step 5: Domain Feature Engineering:</b> Domain transformations (`vehicle_age`, `annual_mileage`, `mileage_vs_expected`, `wear_index`) are executed.", bullet))
    story.append(Paragraph("<b>Step 6: Preprocessing Transformer Fitting:</b> StandardScaler, OrdinalEncoder, and OneHotEncoder are fitted strictly on the training subset.", bullet))
    story.append(Paragraph("<b>Step 7: Log-Target Regressor Fitting:</b> Candidate models are trained on $\\log(1 + y)$ using the transformed training features.", bullet))
    story.append(Paragraph("<b>Step 8: Cross-Validation & Metric Tuning:</b> 5-fold cross-validation verifies stability across subsets.", bullet))
    story.append(Paragraph("<b>Step 9: Holdout Evaluation & Serialization:</b> Final metrics, bootstrap intervals, and SHAP values are extracted on the 3,750 holdout vehicles; the winning LightGBM pipeline is serialized to `models/car_valuation_pipeline.joblib`.", bullet))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 53: UNIT III - 3.5 EXPECTED OUTCOMES, CONCLUSION & REFERENCES
    # =========================================================================
    story.append(Paragraph("3.5 Expected Outcome of Sampling Design", h2))
    story.append(Paragraph("The stratified sampling design guarantees that the resulting 25,000-record dataset provides a balanced, statistically representative cross-section of the used automobile marketplace. Selection bias is eliminated, minority vehicle classes are adequately represented, and model generalization is verified across all price and age brackets.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("OVERALL CONCLUSION", h1))
    story.append(Paragraph("This comprehensive academic project report successfully executes the complete research methodology for <b>Resale Wale</b> in compliance with course <b>24AL58</b>. By combining secondary marketplace data, domain feature engineering, log-target variance stabilization, and gradient-boosted decision trees, the system achieves an out-of-sample R² of 0.9640 and a MAPE of 8.03% (95% CI: [7.84%, 8.22%]). Hypothesis testing decisively rejects the linear null hypothesis ($p < 0.001$). Integrated SHAP explanations and an interactive web terminal translate mathematical precision into practical economic transparency.", body))
    story.append(Spacer(1, 6))
    story.append(Paragraph("REFERENCES (IEEE FORMAT)", h2))
    story.append(Paragraph("[1] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining</i>, 2016, pp. 785–794.", bullet))
    story.append(Paragraph("[2] G. Ke et al., \"LightGBM: A Highly Efficient Gradient Boosting Decision Tree,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 3146–3154.", bullet))
    story.append(Paragraph("[3] S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 4765–4774.", bullet))
    story.append(Paragraph("[4] S. Rosen, \"Hedonic Prices and Implicit Markets: Product Differentiation in Pure Competition,\" <i>Journal of Political Economy</i>, vol. 82, no. 1, pp. 34–55, 1974.", bullet))
    story.append(Paragraph("[5] F. Pedregosa et al., \"Scikit-learn: Machine Learning in Python,\" <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825–2830, 2011.", bullet))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {output_filename}!")

if __name__ == '__main__':
    out_pdf = "RMIPR_REPORT_FINAL_RESALE_WALE.pdf"
    if len(sys.argv) > 1:
        out_pdf = sys.argv[1]
    build_pdf(out_pdf)
