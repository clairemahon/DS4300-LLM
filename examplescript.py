import pandas as pd
import time
import logging
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zero_shot_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("zero_shot_analysis")


class ZeroShotAnalysis:
    """Zero-Shot approach for analyzing gender-based support in Kickstarter."""

    def __init__(self, base_url="http://localhost:5001", model_name="llama3.1:8b"):
        """Initialize the Zero-Shot analysis system."""
        logger.info(
            f"Initializing Zero-Shot Analysis with model: {model_name}")

        # Configure Ollama LLM
        self.llm = OllamaLLM(
            base_url=base_url,
            model=model_name,
            temperature=0.2
        )

    def analyze_gender_support(self, processed_df):
        """Perform Zero-Shot analysis of male vs. female backer support by category."""
        logger.info(
            f"Starting Zero-Shot analysis on {len(processed_df)} projects")

        # Start timing
        start_time = time.time()

        print("\n" + "=" * 80)
        print("🚀 STARTING ZERO-SHOT ANALYSIS 🚀")
        print("=" * 80)

        # Prepare data summary for prompt
        data_summary = self._prepare_data_summary(processed_df)

        # Create comprehensive zero-shot prompt
        zero_shot_prompt = PromptTemplate.from_template("""
            Analyze the Kickstarter dataset to identify patterns in backer support for male versus female creators across different project categories.

            **DATASET SUMMARY:**
            {data_summary}

            **ANALYSIS TASKS:**
            1. Compare backer counts for male vs. female creators across categories.
            2. Identify categories with significant gender disparities in backer support.
            3. Calculate average backer counts by gender for each category.
            4. Determine if there are statistically significant differences in support patterns.
            5. Analyze potential factors influencing gender-based backer support.

            **FORMAT YOUR RESPONSE** as a structured business report with:
            1. **Executive Summary** (key findings in 100 words or less)
            2. **Methodology** (brief explanation of analytical approach)
            3. **Gender Support Analysis by Category** (with supporting data)
            4. **Key Disparities and Patterns** (focus on the most significant findings)
            5. **Business Implications and Recommendations**

            **Include tables** where appropriate to summarize numerical data. Focus on clear, actionable insights.
        """)

        print("\n📊 Running comprehensive gender support analysis...")
        print("-" * 50)
        print("🔄 Sending dataset summary to LLM for processing...")
        time.sleep(2)

        # Execute Zero-Shot analysis
        analysis_response = self.llm.invoke(zero_shot_prompt.format(
            data_summary=data_summary
        ))

        # End timing
        end_time = time.time()
        processing_time = end_time - start_time

        print("\n" + "=" * 80)
        print(
            f"🎯 ZERO-SHOT ANALYSIS COMPLETED IN {processing_time:.2f} SECONDS 🎯")
        print("=" * 80 + "\n")

        # Save analysis to file
        self._save_analysis(analysis_response, processing_time)

        return {
            "analysis": analysis_response,
            "processing_time": processing_time
        }

    def _prepare_data_summary(self, df):
        """Prepare a concise summary of the dataset for the Zero-Shot prompt."""

        # Gender Distribution
        gender_counts = df['likely_gender'].value_counts(normalize=True) * 100
        gender_summary = {gender: f"{count:.2f}%" for gender,
                          count in gender_counts.items()}

        # Top Categories
        top_categories = df['category'].value_counts().head(5).to_dict()

        # Backer Stats by Gender
        gender_backers = df.groupby('likely_gender')['backers_count'].agg([
            'mean', 'median', 'max']).to_dict()

        # Format summary text
        summary = f"""
        **Dataset Overview**
        - Total Projects: {len(df)}

        **Gender Distribution (% of Total Projects)**
        {gender_summary}

        **Top 5 Categories**
        {top_categories}

        **Backer Statistics by Gender**
        {gender_backers}
        """

        return summary.strip()

    def _save_analysis(self, analysis, processing_time):
        """Save Zero-Shot analysis to a markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"zero_shot_gender_support_{timestamp}.md"

        with open(output_file, "w") as f:
            f.write("# Male vs. Female Backer Support Analysis in Kickstarter\n\n")
            f.write(
                f"*Analysis completed on {datetime.now().strftime('%B %d, %Y')} in {processing_time:.2f} seconds*\n\n")
            f.write("## Zero-Shot Analysis Results\n\n")
            f.write(analysis)

        logger.info(f"Analysis saved to {output_file}")


def main():
    """Execute Zero-Shot analysis on Kickstarter data."""
    logger.info("Starting Zero-Shot Analysis demonstration")

    # Load pre-processed data
    try:
        data_file = "Cleaned_Kickstarter_Data_1k.csv"
        print(f"📂 Loading data from {data_file}...")

        processed_df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(processed_df)} records successfully.")

        # Initialize and run analysis
        zero_shot = ZeroShotAnalysis()
        result = zero_shot.analyze_gender_support(processed_df)

        print("\n🎉 Analysis complete!")
        print("📄 Results saved to markdown file.")
        print("\n🔍 This Zero-Shot approach demonstrates:")
        print("1️⃣ Direct, comprehensive analysis without intermediate steps")
        print("2️⃣ Efficient execution through a single, well-crafted prompt")
        print("3️⃣ A focus on the specific insight of male vs. female backer support")

    except Exception as e:
        logger.error(f"❌ Error in execution: {str(e)}", exc_info=True)
        print(f"⚠️ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
