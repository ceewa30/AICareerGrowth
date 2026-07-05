import os
import sys
import shutil
from typing import Any, Tuple
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
# Ensure root paths are resolved correctly
root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))

root_dir = Path(__file__).resolve().parent

env_path = Path(__file__).resolve().parent / ".env"


# Import our core application components
from CareerTrackerAgent import CareerTracker
from rag_comparison_engine import RAGComparisonEngine
from pdf_exporter import export_markdown_to_pdf
from Gaurdrails.IngestionGuardrails import IngestionGuardrails

load_dotenv(dotenv_path=env_path, override=True)

def process_file_upload(file_obj) -> str:
    """Helper to detect file types and parse text out of incoming Gradio file objects safely."""
    print("PDF Loading ....")
    if file_obj is None:
        return ""

    temp_path = Path(file_obj.name)
    file_extension = temp_path.suffix.lower()

    if file_extension not in [".pdf", ".docx"]:
        return "⚠️ Unsupported File Format: Please upload a standard .pdf or .docx resume.", ""

    if (root_dir / "resources").exists():
        project_root = root_dir
    else:
        project_root = root_dir.parent

    output_dir = project_root / "resources"

    target_filename = f"linkedin_profile{file_extension}"
    local_file_path = output_dir / target_filename

    print(f"📁 Saving file locally to: {local_file_path}")

    try:
        # Copy the temporary file to your local directory safely
        shutil.copy(temp_path, local_file_path)
    except Exception as e:
        return f"❌ Storage Error: Failed to save file locally. Detail: {str(e)}", ""

    try:
        parsed_text = ""
        if file_extension == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(local_file_path))
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        elif file_extension == ".docx":
            import docx
            doc = docx.Document(str(local_file_path))
            return "\n".join([p.text for p in doc.paragraphs if p.text])

        if not parsed_text.strip():
            return f"⚠️ Warning: File saved to {local_file_path} but no readable text was found.", ""

        return f"✅ File successfully saved as '{local_file_path.name}' and parsed.", parsed_text
    except Exception as e:
        return f"❌ Ingestion Error: Failed to parse uploaded file. Detail: {str(e)}"


def run_ai_career_pipeline(current_role: str,
                            github_username: str,
                            target_goal: str,
                            uploaded_file: Any,
                            progress=gr.Progress()
                        ) -> Tuple[str, str, str]:
    """
    Reactive wrapper that ties the interface fields directly into your existing backend processing scripts.
    """
    if not current_role.strip() or not github_username.strip() or not target_goal.strip():
        err_html = "<div style='color: #DD6B20; font-weight: bold;'>⚠️ Validation Error: Please fill out all text input boxes before executing.</div>"
        return "### ⚠️ Incomplete Submission", err_html, "Please input your active engineering profile parameters."

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("GITHUB_TOKEN_KEY"):
        env_html = "<div style='color: #E53E3E; font-weight: bold;'>❌ Environment Error: Missing OPENAI_API_KEY or GITHUB_TOKEN inside your local .env config.</div>"
        return "### ❌ System Exception", env_html, "Backend cluster connectivity is currently unavailable."

    try:
        # -------------------------------------------------------------------------
        # PHASE 1: BACKGROUND INGESTION PIPELINE
        # -------------------------------------------------------------------------
        progress(0.05, desc="processing uploaded resume document text tracking layers...")
        resume_extract_text = process_file_upload(uploaded_file)

        if not resume_extract_text:
            print("ℹ️ No file uploaded. Processing pipeline using system database default strings.")
            resume_extract_text = f"Candidate current appointed role title: {current_role}"

        guardrail = IngestionGuardrails()

        is_safe, check_message = guardrail.evaluate_input_safety(resume_extract_text)

        if not is_safe:
            # Short-circuit processing instantly, updating UI blocks with error parameters safely
            fallback_error_card = f"""
            <div style="background-color: #fef2f2; border: 1px solid #f87171; padding: 16px; border-radius: 8px; color: #991b1b;">
                <strong>⚠️ Ingestion Blocked</strong><br>{check_message}
            </div>
            """
            return (
                "### ❌ Processing Terminated\nInput text failed security and validation parameter gates.",
                fallback_error_card,
                "### 🚀 Upskilling Plan Unavailable\nPlease clean the text profile artifact and try again."
            )
        sanitized_resume_content = guardrail.sanitize_raw_text(resume_extract_text)
        print("✅ Ingestion guardrails passed successfully. Commencing vector storage execution tracking.")

        progress(0.3, desc="Initializing Vector Database Collections...")
        tracker = CareerTracker()

        progress(0.5, desc=f"Connecting to GitHub API for user: '{github_username}'...")
        tracker.username = github_username

        progress(0.7, desc="Parsing Resume & Extracting Source Repositories... '{input_target}'")
        tracker.skill_combine()

        # -------------------------------------------------------------------------
        # PHASE 2: SEMANTIC COMPARISON ENGINE
        # -------------------------------------------------------------------------
        progress(0.9, desc="Analyzing Knowledge Gaps via Semantic RAG Query...")
        normalized_role = current_role.lower().replace(' ', '_').replace('/', '_')
        derived_profile_id = f"profile_{normalized_role}"

        engine = RAGComparisonEngine()

        results_dict = engine.generate_career_gap_report(
            target_role=target_goal,
            user_profile_id=derived_profile_id
        )

        # -------------------------------------------------------------------------
        # PHASE 3: CONVERT STRUCTURAL PAYLOAD TO CLEAN VISUAL MARKDOWN
        # -------------------------------------------------------------------------
        progress(0.9, desc="Compiling Career Pathway Report Blueprint...")

        # Pull parameters dynamically with clear dictionary fallbacks
        readiness_score = results_dict.get("readiness_score", 0.0)
        strengths = results_dict.get("current_strengths_detected") or []
        gaps = results_dict.get("mathematically_verified_gaps") or []
        roadmap_markdown_text = results_dict.get("phased_roadmap_text") or "Detailed upskilling path completed."

        strengths_list = "\n".join([f"- ✅ **{s}**" for s in strengths]) if strengths else "* No explicit capability overlaps detected."
        gaps_list = "\n".join([f"- ❌ *{gap}*" for gap in gaps]) if gaps else "* No critical structural skill gaps remaining."

        # This keeps the markdown pristine and ensures Weasyprint/xhtml2pdf won't crash on outer divs later!
        markdown_top = f"""# 🎯 AI Career Pathway Report: {target_goal}
*Generated for current role: **{current_role}** (GitHub: @{github_username})*

---

## 🚀 Identified Strengths
Your existing engineering baseline gives you a strong leverage foundation in these areas:
{strengths_list}
"""
        # The beautiful, native HTML progress card
        metric_banner_html = f"""
        <div style="background-color: #F7FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 11pt; color: #4A5568; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em;">Current Role Readiness Score</p>
            <div style="display: flex; align-items: center; margin-top: 5px;">
                <div style="font-size: 28pt; font-weight: 800; color: #2B6CB0; margin-right: 15px;">{readiness_score}%</div>
                <div style="background-color: #E2E8F0; border-radius: 9999px; width: 100%; height: 12px; overflow: hidden;">
                    <div style="background-color: #3182CE; width: {readiness_score}%; height: 100%; border-radius: 9999px;"></div>
                </div>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 10pt; color: #718096;">Your agent has identified <b>{len(gaps)}</b> critical framework gaps remaining to hit this target.</p>
        </div>
        """
        markdown_bottom = f"""
## 🔍 Target Knowledge Gaps
To secure your desired role, your agent determined you must close these critical gaps:
{gaps_list}

---

## 🛠️ Recommended Sequential Upskilling Roadmap
Follow these phased steps to bridge your technical skills gaps and upgrade your profile:
{roadmap_markdown_text}
"""
        progress(1.0, desc="Report successfully hydrated!")
        # Return elements matching our output layout sequentially
        return markdown_top, metric_banner_html, markdown_bottom

    except Exception as e:
        # Graceful traceback fallback shield map
        import traceback
        print("\n💥 PIPELINE EXCEPTION TRACEBACK:")
        traceback.print_exc()

        # 2. Fix: Always return EXACTLY 3 string elements to satisfy your [markdown, html, markdown] layout components
        error_msg = f"<p style='color:#E53E3E;'>💥 <b>Pipeline Execution Crash:</b> {str(e)}</p>"
        return error_msg, "", ""


def trigger_pdf_generation(cached_markdown: str):
    """Takes the active UI markdown cache state, calls the compiler, and reveals the file download object."""
    if not cached_markdown or "AI Career Pathway Report" not in cached_markdown:
        return gr.update(value=None, visible=False), "⚠️ Error: Please generate a roadmap report first before exporting."

    # Calls your external open-source compiler library
    local_pdf_path = export_markdown_to_pdf(cached_markdown, output_filename="AI_Career_Upskilling_Roadmap.pdf")

    if local_pdf_path:
        return gr.update(value=local_pdf_path, visible=True), "🎉 PDF Document compiled successfully! Click below to download."
    return gr.update(value=None, visible=False), "❌ PDF Generation failed. Verify your console log outputs."

def merge_fields_for_pdf(top_text: str, bottom_text: str):
    """Combines your top and bottom markdown blocks seamlessly into a single long report for the PDF exporter."""
    combined_markdown = f"{top_text}\n\n{bottom_text}"
    return trigger_pdf_generation(combined_markdown)


# =====================================================================
# GRADIO APPLICATION THEME & INTERFACE BUILDER
# =====================================================================
with gr.Blocks(theme=gr.themes.Soft(), title="AI Career Growth Tracker") as app:

    # 🚀 1: Initialize hidden stable background memory blocks to hold text strings
    stored_top_markdown = gr.State(value="")
    stored_bottom_markdown = gr.State(value="")

    gr.Markdown("""
    # 🧠 AI Career Growth Tracker Dashboard
    An autonomous agent that monitors job trends, extracts engineering capabilities, and maps strict certification paths to hit your target goals.
    """)

    with gr.Row():
        # Left Workspace: Input controls
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Candidate Parameters Input")
            input_role = gr.Textbox(label="Current Role Title", value="Full Stack Developer")
            input_github = gr.Textbox(label="GitHub Username", value="octocat")
            input_target = gr.Textbox(label="Aspirational Target Role", value="AI Engineer / Data Science")
            input_file = gr.File(
                label="Upload Current Resume Document",
                file_types=[".pdf", ".docx"],
                type="filepath"
            )
            submit_btn = gr.Button("⚡ Generate Upskilling Roadmap", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### 📥 Export Capabilities")
            export_btn = gr.Button("📄 Compile & Export PDF Report", variant="secondary")
            status_ticker = gr.Markdown(value="")
            file_download_module = gr.File(label="Download Compiled PDF Report", visible=False)

        # Right Workspace: Output display
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Automated Coaching Output")
            output_strengths_display = gr.Markdown(value="*Your strengths profile will appear here.*")
            html_metric_card = gr.HTML(value="<div style='color:#718096; font-style:italic;'>Metrics uninitialized.</div>")
            output_roadmap_display = gr.Markdown(value="*Your custom upskilling roadmap phases will appear here.*")

    # 🚀 2: Stream outputs to your dashboard AND capture the text inputs inside your gr.State cells
    submit_btn.click(
        fn=run_ai_career_pipeline,
        inputs=[input_role, input_github, input_target, input_file],
        outputs=[output_strengths_display, html_metric_card, output_roadmap_display]
    ).then(
        # This lambda function safely copies the screen outputs into your hidden memory buffers
        fn=lambda top, bottom: (top, bottom),
        inputs=[output_strengths_display, output_roadmap_display],
        outputs=[stored_top_markdown, stored_bottom_markdown]
    )

    # 🚀 3: Pull text strings securely from State cells rather than unstable component configurations
    export_btn.click(
        fn=merge_fields_for_pdf,
        inputs=[stored_top_markdown, stored_bottom_markdown],
        outputs=[file_download_module, status_ticker]
    )

if __name__ == "__main__":
    allowed_export_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "exports"))
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[allowed_export_folder],
        theme=gr.themes.Soft()
    )
