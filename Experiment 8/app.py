from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "Experiment 6" / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


st.title("Employee Attrition Prediction Dashboard")

st.write(
    "An interactive machine learning dashboard for employee attrition "
    "prediction, model evaluation, explainability, drift monitoring, "
    "and Responsible AI analysis."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Overview",
        "Attrition Predictor",
        "Model Performance",
        "SHAP Explainability",
        "Data Drift",
        "Responsible AI",
    ],
)


if page == "Overview":
    st.header("Employee Attrition Prediction")

    st.write(
        "This project applies machine learning to employee-related attributes "
        "to estimate whether an employee is likely to stay with or leave "
        "the organization."
    )

    st.subheader("Final Model")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Random Forest")
    col2.metric("Input Features", "44")
    col3.metric("Target", "Turnover Status")

    st.subheader("Dashboard Modules")

    st.write(
        """
        - **Attrition Predictor:** Generate predictions for individual employee profiles.
        - **Model Performance:** Review evaluation metrics of the trained model.
        - **SHAP Explainability:** Examine factors influencing model predictions.
        - **Data Drift:** Monitor changes in employee feature distributions.
        - **Responsible AI:** Review fairness, privacy, transparency, and responsible-use considerations.
        """
    )

    st.info(
        "This system is intended for educational and analytical purposes. "
        "Predictions should support human decision-making and should not be "
        "used as the sole basis for employment decisions."
    )


elif page == "Attrition Predictor":
    st.header("Real-Time Employee Attrition Predictor")

    st.write(
        "Enter employee information below to generate an attrition prediction "
        "using the trained Random Forest pipeline."
    )

    with st.form("prediction_form"):

        st.subheader("Personal and Employment Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 18, 70, 32)
            gender = st.selectbox("Gender", ["Female", "Male"])
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
            )
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Diploma", "Bachelor", "Master", "PhD"],
                index=2,
            )
            department = st.selectbox(
                "Department",
                ["IT", "HR", "Finance", "Sales", "Operations", "Marketing"],
            )

        with c2:
            job_role = st.text_input("Job Role", "Software Engineer")
            job_level = st.number_input("Job Level", 1, 5, 2)
            employment_type = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract"],
            )
            work_mode = st.selectbox(
                "Work Mode",
                ["On-site", "Hybrid", "Remote"],
                index=1,
            )
            monthly_income = st.number_input(
                "Monthly Income",
                min_value=0.0,
                value=55000.0,
            )

        with c3:
            salary_hike_pct = st.number_input(
                "Salary Hike (%)",
                min_value=0.0,
                value=10.0,
            )
            bonus_amount = st.number_input(
                "Bonus Amount",
                min_value=0.0,
                value=5000.0,
            )
            stock_option_level = st.number_input(
                "Stock Option Level",
                0,
                5,
                1,
            )
            performance_rating = st.number_input(
                "Performance Rating",
                1,
                5,
                4,
            )
            projects_completed = st.number_input(
                "Projects Completed",
                min_value=0,
                value=8,
            )

        st.subheader("Attendance and Productivity")

        c1, c2, c3 = st.columns(3)

        with c1:
            productivity_score = st.number_input(
                "Productivity Score",
                0.0,
                100.0,
                82.0,
            )
            working_days = st.number_input(
                "Working Days",
                min_value=0,
                value=250,
            )
            absent_days = st.number_input(
                "Absent Days",
                min_value=0,
                value=5,
            )

        with c2:
            late_arrival_days = st.number_input(
                "Late Arrival Days",
                min_value=0,
                value=3,
            )
            attendance_rate = st.number_input(
                "Attendance Rate (%)",
                0.0,
                100.0,
                96.0,
            )
            weekly_work_hours = st.number_input(
                "Weekly Work Hours",
                min_value=0.0,
                value=40.0,
            )

        with c3:
            overtime = st.selectbox("Overtime", ["No", "Yes"])
            business_travel_frequency = st.selectbox(
                "Business Travel Frequency",
                ["Never", "Rarely", "Frequently"],
                index=1,
            )
            distance_from_home_km = st.number_input(
                "Distance From Home (km)",
                min_value=0.0,
                value=10.0,
            )

        st.subheader("Experience and Career Development")

        c1, c2, c3 = st.columns(3)

        with c1:
            years_at_company = st.number_input(
                "Years at Company",
                min_value=0,
                value=4,
            )
            years_in_current_role = st.number_input(
                "Years in Current Role",
                min_value=0,
                value=2,
            )
            years_since_last_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                value=1,
            )
            promotion_count = st.number_input(
                "Promotion Count",
                min_value=0,
                value=1,
            )

        with c2:
            training_hours_year = st.number_input(
                "Training Hours per Year",
                min_value=0.0,
                value=40.0,
            )
            training_programs_attended = st.number_input(
                "Training Programs Attended",
                min_value=0,
                value=3,
            )
            skill_development_score = st.number_input(
                "Skill Development Score",
                0.0,
                100.0,
                80.0,
            )
            career_growth_opportunity = st.number_input(
                "Career Growth Opportunity",
                1,
                5,
                4,
            )

        with c3:
            experience_group = st.selectbox(
                "Experience Group",
                ["Entry", "Mid", "Senior"],
                index=1,
            )
            tenure_group = st.selectbox(
                "Tenure Group",
                ["Short", "Medium", "Long"],
                index=1,
            )

        st.subheader("Satisfaction and Engagement")

        c1, c2, c3 = st.columns(3)

        with c1:
            job_satisfaction = st.number_input(
                "Job Satisfaction",
                1,
                5,
                4,
            )
            environment_satisfaction = st.number_input(
                "Environment Satisfaction",
                1,
                5,
                4,
            )
            relationship_satisfaction = st.number_input(
                "Relationship Satisfaction",
                1,
                5,
                4,
            )

        with c2:
            manager_satisfaction = st.number_input(
                "Manager Satisfaction",
                1,
                5,
                4,
            )
            work_life_balance = st.number_input(
                "Work-Life Balance",
                1,
                5,
                3,
            )
            overall_satisfaction = st.number_input(
                "Overall Satisfaction",
                1.0,
                5.0,
                4.0,
            )

        with c3:
            employee_engagement_score = st.number_input(
                "Employee Engagement Score",
                0.0,
                100.0,
                80.0,
            )
            job_involvement = st.number_input(
                "Job Involvement",
                1,
                5,
                4,
            )
            recognition_score = st.number_input(
                "Recognition Score",
                0.0,
                100.0,
                75.0,
            )
            company_support_score = st.number_input(
                "Company Support Score",
                0.0,
                100.0,
                80.0,
            )

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:

        employee_data = {
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education_level": education_level,
            "department": department,
            "job_role": job_role,
            "job_level": job_level,
            "employment_type": employment_type,
            "work_mode": work_mode,
            "monthly_income": monthly_income,
            "salary_hike_pct": salary_hike_pct,
            "bonus_amount": bonus_amount,
            "stock_option_level": stock_option_level,
            "performance_rating": performance_rating,
            "projects_completed": projects_completed,
            "productivity_score": productivity_score,
            "working_days": working_days,
            "absent_days": absent_days,
            "late_arrival_days": late_arrival_days,
            "attendance_rate": attendance_rate,
            "years_at_company": years_at_company,
            "years_in_current_role": years_in_current_role,
            "years_since_last_promotion": years_since_last_promotion,
            "promotion_count": promotion_count,
            "training_hours_year": training_hours_year,
            "training_programs_attended": training_programs_attended,
            "skill_development_score": skill_development_score,
            "job_satisfaction": job_satisfaction,
            "environment_satisfaction": environment_satisfaction,
            "relationship_satisfaction": relationship_satisfaction,
            "manager_satisfaction": manager_satisfaction,
            "work_life_balance": work_life_balance,
            "overtime": overtime,
            "weekly_work_hours": weekly_work_hours,
            "business_travel_frequency": business_travel_frequency,
            "employee_engagement_score": employee_engagement_score,
            "job_involvement": job_involvement,
            "recognition_score": recognition_score,
            "distance_from_home_km": distance_from_home_km,
            "company_support_score": company_support_score,
            "career_growth_opportunity": career_growth_opportunity,
            "experience_group": experience_group,
            "tenure_group": tenure_group,
            "overall_satisfaction": overall_satisfaction,
        }

        input_df = pd.DataFrame([employee_data])

        try:
            prediction = int(model.predict(input_df)[0])

            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(input_df)[0][1])
            else:
                probability = None

            st.divider()
            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Prediction: Employee is likely to leave")
            else:
                st.success("Prediction: Employee is likely to stay")

            if probability is not None:
                st.metric(
                    "Estimated Attrition Probability",
                    f"{probability * 100:.2f}%",
                )
                st.progress(min(max(probability, 0.0), 1.0))

            st.caption(
                "This prediction is a model-generated estimate and should "
                "not be used as the sole basis for employment decisions."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


elif page == "Model Performance":
    st.header("Model Performance")

    st.write(
        "The final Random Forest model was selected for deployment based "
        "on the evaluation performed during model development."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", "95.26%")
    c2.metric("Precision", "92.96%")
    c3.metric("Recall", "99.33%")
    c4.metric("F1 Score", "96.04%")
    c5.metric("ROC-AUC", "98.14%")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ],
            "Score": [
                0.9526,
                0.9296,
                0.9933,
                0.9604,
                0.9814,
            ],
        }
    )

    st.subheader("Evaluation Summary")
    st.dataframe(metrics_df, use_container_width=True)

    st.bar_chart(
        metrics_df.set_index("Metric"),
        y="Score",
    )


elif page == "SHAP Explainability":
    st.header("SHAP Explainability")

    st.write(
        "SHAP (SHapley Additive exPlanations) is used to explain how "
        "individual features influence predictions made by the trained "
        "Random Forest model."
    )

    st.info(
        "The explanation below uses a representative employee profile "
        "and evaluates the contribution of the transformed model features "
        "toward the attrition prediction."
    )

    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        sample_employee = {
            "age": 32,
            "gender": "Female",
            "marital_status": "Single",
            "education_level": "Bachelor",
            "department": "IT",
            "job_role": "Software Engineer",
            "job_level": 2,
            "employment_type": "Full-time",
            "work_mode": "Hybrid",
            "monthly_income": 55000,
            "salary_hike_pct": 10,
            "bonus_amount": 5000,
            "stock_option_level": 1,
            "performance_rating": 4,
            "projects_completed": 8,
            "productivity_score": 82,
            "working_days": 250,
            "absent_days": 5,
            "late_arrival_days": 3,
            "attendance_rate": 96,
            "years_at_company": 4,
            "years_in_current_role": 2,
            "years_since_last_promotion": 1,
            "promotion_count": 1,
            "training_hours_year": 40,
            "training_programs_attended": 3,
            "skill_development_score": 80,
            "job_satisfaction": 4,
            "environment_satisfaction": 4,
            "relationship_satisfaction": 4,
            "manager_satisfaction": 4,
            "work_life_balance": 3,
            "overtime": "No",
            "weekly_work_hours": 40,
            "business_travel_frequency": "Rarely",
            "employee_engagement_score": 80,
            "job_involvement": 4,
            "recognition_score": 75,
            "distance_from_home_km": 10,
            "company_support_score": 80,
            "career_growth_opportunity": 4,
            "experience_group": "Mid",
            "tenure_group": "Medium",
            "overall_satisfaction": 4,
        }

        sample_df = pd.DataFrame([sample_employee])

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed_sample = preprocessor.transform(sample_df)

        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed_sample)

        if isinstance(shap_values, list):
            attrition_values = np.asarray(shap_values[1])[0]
        else:
            values = np.asarray(shap_values)

            if values.ndim == 3:
                attrition_values = values[0, :, 1]
            elif values.ndim == 2:
                attrition_values = values[0]
            else:
                attrition_values = values

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": attrition_values,
                "Absolute SHAP Value": np.abs(attrition_values),
            }
        )

        importance_df = importance_df.sort_values(
            "Absolute SHAP Value",
            ascending=False,
        ).head(15)

        st.subheader("Top Features Influencing Attrition")

        display_df = importance_df[
            ["Feature", "SHAP Value", "Absolute SHAP Value"]
        ].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("SHAP Feature Contribution Plot")

        plot_df = importance_df.sort_values(
            "Absolute SHAP Value",
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            plot_df["Feature"],
            plot_df["SHAP Value"],
        )

        ax.axvline(0, linewidth=1)

        ax.set_xlabel("SHAP Value")
        ax.set_ylabel("Feature")
        ax.set_title(
            "Feature Contributions to Attrition Prediction"
        )

        plt.tight_layout()

        st.pyplot(fig)

        prediction = int(model.predict(sample_df)[0])
        probability = float(model.predict_proba(sample_df)[0][1])

        st.subheader("Explained Prediction")

        c1, c2 = st.columns(2)

        c1.metric(
            "Prediction",
            "Left" if prediction == 1 else "Stayed",
        )

        c2.metric(
            "Attrition Probability",
            f"{probability * 100:.2f}%",
        )

        st.caption(
            "Positive SHAP values push the model toward the attrition "
            "class, while negative SHAP values push the prediction toward "
            "the stay class."
        )

    except Exception as error:
        st.error(f"Unable to generate SHAP explanation: {error}")


elif page == "Data Drift":
    st.header("Data Drift Monitoring")

    st.write(
        "Data drift occurs when the statistical distribution of incoming "
        "employee data changes compared with the data used as a reference. "
        "Such changes may affect model reliability over time."
    )

    st.info(
        "Upload a reference dataset and a current employee dataset to "
        "perform a statistical drift comparison. No drift result is "
        "reported until both datasets are provided."
    )

    from scipy.stats import ks_2samp

    col1, col2 = st.columns(2)

    with col1:
        reference_file = st.file_uploader(
            "Upload Reference Dataset",
            type=["csv"],
            key="reference_dataset",
        )

    with col2:
        current_file = st.file_uploader(
            "Upload Current Dataset",
            type=["csv"],
            key="current_dataset",
        )

    if reference_file is None or current_file is None:
        st.subheader("Drift Check Status")

        st.warning(
            "Waiting for both datasets. A valid drift comparison requires "
            "reference data and current data."
        )

        st.write(
            "**Method:** Two-sample Kolmogorov-Smirnov (KS) test for "
            "common numerical features."
        )

        st.write(
            "**Interpretation:** A p-value below 0.05 is treated as a "
            "statistical signal that the two feature distributions may "
            "differ."
        )

    else:
        try:
            reference_df = pd.read_csv(reference_file)
            current_df = pd.read_csv(current_file)

            st.subheader("Dataset Summary")

            c1, c2 = st.columns(2)

            c1.metric(
                "Reference Records",
                f"{len(reference_df):,}",
            )

            c2.metric(
                "Current Records",
                f"{len(current_df):,}",
            )

            common_columns = [
                column
                for column in reference_df.columns
                if column in current_df.columns
            ]

            numerical_columns = [
                column
                for column in common_columns
                if pd.api.types.is_numeric_dtype(reference_df[column])
                and pd.api.types.is_numeric_dtype(current_df[column])
            ]

            if not numerical_columns:
                st.error(
                    "No common numerical features were found between "
                    "the two datasets."
                )

            else:
                drift_results = []

                for column in numerical_columns:
                    reference_values = (
                        reference_df[column]
                        .dropna()
                        .astype(float)
                    )

                    current_values = (
                        current_df[column]
                        .dropna()
                        .astype(float)
                    )

                    if (
                        len(reference_values) == 0
                        or len(current_values) == 0
                    ):
                        continue

                    statistic, p_value = ks_2samp(
                        reference_values,
                        current_values,
                    )

                    drift_results.append(
                        {
                            "Feature": column,
                            "KS Statistic": statistic,
                            "P-Value": p_value,
                            "Statistical Signal": (
                                "Potential Drift"
                                if p_value < 0.05
                                else "No Significant Signal"
                            ),
                        }
                    )

                if not drift_results:
                    st.warning(
                        "The datasets did not contain enough numerical "
                        "values for the drift test."
                    )

                else:
                    drift_df = pd.DataFrame(drift_results)

                    drift_df = drift_df.sort_values(
                        "KS Statistic",
                        ascending=False,
                    )

                    drift_count = (
                        drift_df["Statistical Signal"]
                        == "Potential Drift"
                    ).sum()

                    tested_count = len(drift_df)

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Features Tested",
                        tested_count,
                    )

                    c2.metric(
                        "Potential Drift Signals",
                        int(drift_count),
                    )

                    c3.metric(
                        "No Significant Signal",
                        int(tested_count - drift_count),
                    )

                    st.subheader("Drift Analysis")

                    st.dataframe(
                        drift_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("KS Statistic by Feature")

                    chart_df = (
                        drift_df[
                            ["Feature", "KS Statistic"]
                        ]
                        .set_index("Feature")
                    )

                    st.bar_chart(chart_df)

                    st.caption(
                        "The KS test identifies statistical differences "
                        "between distributions. A detected signal does not "
                        "by itself prove harmful model drift and should be "
                        "reviewed together with model performance and "
                        "operational context."
                    )

        except Exception as error:
            st.error(
                f"Unable to perform drift analysis: {error}"
            )


elif page == "Responsible AI":
    st.header("Responsible AI Assessment")

    st.write(
        "Employee attrition prediction involves workforce-related data and "
        "can affect individuals if predictions are used in organizational "
        "decision-making. Responsible AI practices are therefore necessary "
        "throughout the system lifecycle."
    )

    st.subheader("Responsible AI Checklist")

    responsible_ai = pd.DataFrame(
        {
            "Area": [
                "Fairness",
                "Privacy",
                "Consent",
                "Transparency",
                "Human Oversight",
                "Explainability",
                "Data Drift",
                "Security",
            ],
            "Project Approach": [
                "Model behavior should be evaluated across relevant employee groups before consequential use.",
                "Only data required for attrition analysis should be collected and access should be restricted.",
                "Organizations deploying the system should establish appropriate notice and consent procedures for employee data.",
                "The dashboard displays model purpose, probabilities, limitations, and responsible-use information.",
                "Predictions are decision-support information and must not independently determine employment actions.",
                "SHAP explanations are provided to examine factors contributing to model predictions.",
                "The dashboard provides statistical drift checking using reference and current datasets.",
                "Model files, employee datasets, API access, and deployment credentials should be protected.",
            ],
            "Status": [
                "Requires Ongoing Evaluation",
                "Required",
                "Deployment Requirement",
                "Implemented",
                "Required",
                "Implemented",
                "Implemented",
                "Deployment Requirement",
            ],
        }
    )

    st.dataframe(
        responsible_ai,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Fairness")

    st.write(
        "Employee characteristics and workplace conditions may differ "
        "between groups. Model performance should therefore be evaluated "
        "across relevant groups before the system is used for decisions "
        "that could affect employees."
    )

    st.warning(
        "No numerical fairness score is reported in this dashboard because "
        "the original evaluation dataset is not currently available for a "
        "group-level fairness audit."
    )

    st.subheader("Privacy and Consent")

    st.write(
        "Employee information can contain personal or sensitive workplace "
        "data. A real deployment should apply data minimization, controlled "
        "access, appropriate retention policies, and organizational "
        "procedures for notice and consent."
    )

    st.subheader("Transparency and Explainability")

    st.write(
        "The dashboard reports prediction probabilities and provides SHAP "
        "feature contributions so that users can inspect factors influencing "
        "the model rather than treating its output as an unexplained result."
    )

    st.subheader("Human Oversight")

    st.error(
        "The model must not be used as the sole basis for hiring, firing, "
        "promotion, disciplinary, or other consequential employment decisions."
    )

    st.write(
        "Predictions indicate statistical patterns learned by the model. "
        "They do not establish an employee's intention to leave and should "
        "be reviewed together with appropriate human judgment and context."
    )

    st.subheader("Monitoring")

    st.write(
        "Model behavior should be monitored after deployment. The Data Drift "
        "module supports comparison of reference and current numerical "
        "feature distributions using the two-sample Kolmogorov-Smirnov test."
    )

    st.info(
        "Responsible AI is an ongoing process. Fairness, privacy, security, "
        "performance, and data quality should be reviewed throughout the "
        "lifecycle of the system."
    )