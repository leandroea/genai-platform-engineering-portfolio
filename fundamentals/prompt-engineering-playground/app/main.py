"""
Main Streamlit application for Prompt Engineering Playground.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from datetime import datetime

from app import __version__
from app import database
from app import llm_service
from app import utils
from app.models import LLMParameters


# Page configuration
st.set_page_config(
    page_title="Prompt Engineering Playground",
    page_icon="🧪",
    layout="wide"
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "current_response" not in st.session_state:
        st.session_state.current_response = None
    if "current_experiment_id" not in st.session_state:
        st.session_state.current_experiment_id = None
    if "comparison_responses" not in st.session_state:
        st.session_state.comparison_responses = []


def render_sidebar():
    """Render the sidebar with navigation and LLM parameters."""
    st.sidebar.title("🧪 Prompt Playground")

    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Test Prompt", "Compare Prompts", "History"]
    )

    # LLM Parameters
    params = utils.render_llm_parameters_sidebar()

    # Model info
    st.sidebar.header("Model Info")
    model_name = "Llama 3.3 70B"
    st.sidebar.info(f"**Model:** {model_name}")
    st.sidebar.info(f"**Total Experiments:** {database.get_experiment_count()}")

    return page, params


def page_test_prompt(parameters: LLMParameters):
    """Test a single prompt."""
    st.header("Test Prompt")

    # Prompt input
    prompt = st.text_area(
        "Enter your prompt:",
        height=200,
        placeholder="Enter your prompt here..."
    )

    # System message (optional)
    system_message = st.text_area(
        "System message (optional):",
        height=100,
        placeholder="Set the context or persona for the model..."
    )

    # Experiment name (optional)
    experiment_name = st.text_input(
        "Experiment name (optional):",
        placeholder="My experiment"
    )

    # Run button
    if st.button("Run Prompt", type="primary", use_container_width=True):
        if not prompt.strip():
            utils.show_error("Please enter a prompt")
            return

        with st.spinner("Calling LLM..."):
            try:
                response = llm_service.call_llm(
                    prompt=prompt,
                    system_message=system_message if system_message.strip() else None,
                    parameters=parameters
                )

                # Store in session state
                st.session_state.current_response = response

                # Save to database
                experiment_id = database.save_experiment(
                    prompt=prompt,
                    parameters=parameters,
                    response=response,
                    experiment_type="single",
                    name=experiment_name if experiment_name.strip() else None
                )
                st.session_state.current_experiment_id = experiment_id

                utils.show_success("Prompt executed successfully!")

            except Exception as e:
                utils.show_error(f"Error calling LLM: {str(e)}")
                return

    # Display response
    if st.session_state.current_response:
        st.divider()
        st.subheader("Response:")

        response_container = st.container()
        with response_container:
            st.markdown(st.session_state.current_response)

        # Evaluation section
        st.divider()
        st.subheader("Evaluate Output")

        if st.session_state.current_experiment_id:
            experiment = database.get_experiment_by_id(st.session_state.current_experiment_id)

            # Rating
            col1, col2 = st.columns([1, 2])
            with col1:
                rating = st.radio(
                    "Rating:",
                    options=[1, 2, 3, 4, 5],
                    horizontal=True,
                    index=(experiment.rating - 1) if experiment.rating else None,
                    key="rating_single"
                )

            with col2:
                feedback = st.text_area(
                    "Feedback (optional):",
                    value=experiment.feedback or "",
                    placeholder="Add your feedback here...",
                    key="feedback_single"
                )

            if st.button("Save Evaluation", use_container_width=True):
                database.update_experiment_rating(
                    st.session_state.current_experiment_id,
                    rating=rating,
                    feedback=feedback if feedback.strip() else None
                )
                utils.show_success("Evaluation saved!")

            # Copy button
            if st.button("Copy Response", use_container_width=True):
                st.code(st.session_state.current_response, language="text")
                utils.show_success("Response copied to clipboard!")


def page_compare_prompts(parameters: LLMParameters):
    """Compare multiple prompts."""
    st.header("Compare Prompts")

    # Number of prompts
    num_prompts = st.slider("Number of prompts to compare", min_value=2, max_value=4, value=2)

    # Prompt inputs
    prompts = []
    for i in range(num_prompts):
        prompt = st.text_area(
            f"Prompt {i + 1}:",
            height=120,
            key=f"compare_prompt_{i}",
            placeholder=f"Enter prompt {i + 1}..."
        )
        prompts.append(prompt)

    # Experiment name
    experiment_name = st.text_input(
        "Experiment name (optional):",
        placeholder="My comparison experiment"
    )

    # Run button
    if st.button("Run Comparison", type="primary", use_container_width=True):
        # Validate prompts
        empty_prompts = [i + 1 for i, p in enumerate(prompts) if not p.strip()]
        if empty_prompts:
            utils.show_error(f"Please fill in prompts: {', '.join(map(str, empty_prompts))}")
            return

        with st.spinner("Calling LLM for all prompts..."):
            try:
                responses = []
                for i, prompt in enumerate(prompts):
                    response = llm_service.call_llm(
                        prompt=prompt,
                        parameters=parameters
                    )
                    responses.append(response)

                # Store in session state
                st.session_state.comparison_responses = responses

                # Save to database
                experiment_id = database.save_experiment(
                    prompt="\n---\n".join(prompts),
                    parameters=parameters,
                    response="Comparison experiment",
                    experiment_type="comparison",
                    name=experiment_name if experiment_name.strip() else None
                )

                # Save individual comparison results
                for i, response in enumerate(responses):
                    database.save_comparison_result(
                        experiment_id=experiment_id,
                        prompt_index=i,
                        response=response
                    )

                st.session_state.current_experiment_id = experiment_id
                utils.show_success("Comparison completed!")

            except Exception as e:
                utils.show_error(f"Error calling LLM: {str(e)}")
                return

    # Display responses
    if st.session_state.comparison_responses:
        st.divider()
        st.subheader("Responses:")

        cols = st.columns(len(st.session_state.comparison_responses))

        for i, (col, response) in enumerate(zip(cols, st.session_state.comparison_responses)):
            with col:
                st.markdown(f"**Prompt {i + 1}**")
                st.markdown(f"_{utils.truncate_text(prompts[i], 50)}_")
                st.markdown("---")
                st.markdown(response)

        # Evaluation section
        st.divider()
        st.subheader("Compare & Evaluate")

        if st.session_state.current_experiment_id:
            comparison_results = database.get_comparison_results(st.session_state.current_experiment_id)

            for i, result in enumerate(comparison_results):
                col1, col2 = st.columns([1, 2])
                with col1:
                    rating = st.radio(
                        f"Prompt {i + 1} Rating:",
                        options=[1, 2, 3, 4, 5],
                        horizontal=True,
                        index=(result.rating - 1) if result.rating else None,
                        key=f"rating_compare_{i}"
                    )

                with col2:
                    if st.button(f"Save Rating {i + 1}", key=f"save_rating_{i}"):
                        database.update_comparison_result_rating(result.id, rating)
                        utils.show_success(f"Rating saved for Prompt {i + 1}!")


def page_history():
    """View experiment history."""
    st.header("Experiment History")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_type = st.selectbox(
            "Filter by type:",
            ["All", "Single", "Comparison"]
        )

    with col2:
        min_rating = st.selectbox(
            "Minimum rating:",
            ["All", "1 ⭐", "2 ⭐", "3 ⭐", "4 ⭐", "5 ⭐"]
        )

    with col3:
        search_text = st.text_input("Search:", placeholder="Search prompts...")

    # Convert filters
    experiment_type = None if filter_type == "All" else filter_type.lower()
    min_rating_val = None if min_rating == "All" else int(min_rating[0])
    search_val = search_text if search_text.strip() else None

    # Get experiments
    experiments = database.get_all_experiments(
        experiment_type=experiment_type,
        min_rating=min_rating_val,
        search_text=search_val
    )

    # Display experiments
    if not experiments:
        st.info("No experiments found. Start testing prompts!")
        return

    st.write(f"Found {len(experiments)} experiments")

    for experiment in experiments:
        with st.expander(f"{utils.format_datetime(experiment.created_at)} - {experiment.name or 'Unnamed'}"):
            # Header info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", experiment.experiment_type.capitalize())
            with col2:
                st.metric("Rating", utils.render_stars(experiment.rating or 0) if experiment.rating else "Not rated")
            with col3:
                if st.button("Delete", key=f"delete_{experiment.id}"):
                    database.delete_experiment(experiment.id)
                    utils.show_success("Experiment deleted!")
                    st.rerun()

            # Prompt
            st.markdown("**Prompt:**")
            st.text_area(
                "Prompt",
                value=experiment.prompt,
                height=100,
                disabled=True,
                key=f"prompt_{experiment.id}"
            )

            # Response (for single experiments)
            if experiment.experiment_type == "single" and experiment.response:
                st.markdown("**Response:**")
                st.text_area(
                    "Response",
                    value=experiment.response,
                    height=150,
                    disabled=True,
                    key=f"response_{experiment.id}"
                )

            # Parameters
            st.markdown("**Parameters:**")
            params = experiment.parameters
            st.code(
                f"temperature={params.temperature}, max_tokens={params.max_tokens}, "
                f"top_p={params.top_p}, frequency_penalty={params.frequency_penalty}, "
                f"presence_penalty={params.presence_penalty}",
                language="python"
            )

            # Feedback
            if experiment.feedback:
                st.markdown("**Feedback:**")
                st.info(experiment.feedback)

            # Re-run button
            if st.button("Re-run Experiment", key=f"rerun_{experiment.id}"):
                st.session_state.rerun_prompt = experiment.prompt
                st.session_state.rerun_params = experiment.parameters


def main():
    """Main application entry point."""
    # Initialize
    init_session_state()
    database.init_db()

    # Render sidebar and get navigation
    page, parameters = render_sidebar()

    # Render header
    st.title("🧪 Prompt Engineering Playground")
    st.markdown(f"*Version {__version__} - Test, compare, and evaluate your prompts*")
    st.divider()

    # Render the selected page
    if page == "Test Prompt":
        page_test_prompt(parameters)
    elif page == "Compare Prompts":
        page_compare_prompts(parameters)
    elif page == "History":
        page_history()


if __name__ == "__main__":
    main()
