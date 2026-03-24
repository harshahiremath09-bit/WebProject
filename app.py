"""
Enterprise Credit Risk & Loan Portfolio Intelligence Platform

Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-header h1 { color: #ffffff; margin: 0; }
    .main-header p { color: #94a3b8; margin: 5px 0 0 0; }
    .stMetric {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 15px; border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False, 'model_trained': False, 'portfolio_df': None,
        'portfolio_metrics': {}, 'model_trainer': None, 'shap_explainer': None,
        'data_source': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_demo_data():
    """Generate synthetic demo data."""
    np.random.seed(42)
    n = 10000
    grades = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n, 
                              p=[0.08, 0.18, 0.25, 0.22, 0.15, 0.08, 0.04])
    grade_pd_base = {'A': 0.05, 'B': 0.10, 'C': 0.15, 'D': 0.22, 'E': 0.30, 'F': 0.40, 'G': 0.50}
    
    df = pd.DataFrame({
        'loan_amnt': np.random.uniform(5000, 40000, n),
        'grade': grades,
        'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 
                                     'major_purchase', 'small_business'], n),
        'home_ownership': np.random.choice(['OWN', 'MORTGAGE', 'RENT'], n, p=[0.15, 0.45, 0.40]),
        'term': np.random.choice([36, 60], n, p=[0.72, 0.28]),
        'annual_inc': np.random.lognormal(11, 0.5, n),
        'dti': np.random.uniform(5, 40, n),
        'int_rate': np.random.uniform(6, 28, n),
    })
    df['pd'] = df['grade'].map(grade_pd_base) + np.random.normal(0, 0.05, n)
    df['pd'] = df['pd'].clip(0.01, 0.95)
    return df


def load_kaggle_data(n_rows=50000):
    """Download and load actual Lending Club data from Kaggle."""
    try:
        import kagglehub
        import tempfile
        import subprocess
        import io
        
        progress = st.progress(0, text="Checking Kaggle dataset cache...")
        
        # Step 1: Download (may be cached)
        progress.progress(10, text="📥 Downloading from Kaggle (may be cached)...")
        path = kagglehub.dataset_download("wordsforthewise/lending-club")
        
        # Step 2: Find CSV
        progress.progress(25, text="🔍 Finding loan data file...")
        csv_files = list(Path(path).glob("*.csv"))
        if not csv_files:
            st.error("No CSV files found in dataset")
            return None
        
        data_file = None
        for f in csv_files:
            if "accepted" in str(f).lower():
                data_file = f
                break
        if not data_file:
            data_file = csv_files[0]
        
        # Step 3: Copy using Windows copy command (bypasses locks)
        progress.progress(35, text="📋 Copying data file...")
        temp_dir = Path(tempfile.gettempdir()) / "credit_risk_data"
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"lending_club_{n_rows}.csv"
        
        try:
            # Use Windows copy command to bypass file locks
            result = subprocess.run(
                ['cmd', '/c', 'copy', '/Y', str(data_file), str(temp_file)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise Exception(f"Copy failed: {result.stderr}")
        except Exception as copy_err:
            st.warning(f"Copy method failed: {copy_err}. Trying direct read...")
            # Fallback: read file in binary mode with sharing
            progress.progress(40, text="📖 Reading file directly...")
            with open(data_file, 'rb') as f:
                content = f.read()
            with open(temp_file, 'wb') as f:
                f.write(content)
        
        # Step 4: Load CSV from temp
        progress.progress(50, text=f"📄 Loading {n_rows:,} rows from CSV...")
        df = pd.read_csv(temp_file, nrows=n_rows, low_memory=False)
        st.success(f"✅ Loaded {len(df):,} records")
        
        # Step 5: Preprocess
        progress.progress(70, text="⚙️ Preprocessing data...")
        from src.data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(df)
        processed_df = preprocessor.get_processed_dataframe()
        
        # Step 6: Generate PD
        progress.progress(90, text="🎯 Calculating risk scores...")
        grade_pd = {'A': 0.05, 'B': 0.10, 'C': 0.18, 'D': 0.25, 'E': 0.35, 'F': 0.45, 'G': 0.55}
        if 'grade' in processed_df.columns:
            processed_df['pd'] = processed_df['grade'].map(grade_pd).fillna(0.2) + np.random.normal(0, 0.03, len(processed_df))
            processed_df['pd'] = processed_df['pd'].clip(0.01, 0.95)
        else:
            processed_df['pd'] = y.values * 0.7 + 0.1
        
        progress.progress(100, text="✅ Complete!")
        return processed_df
        
    except PermissionError as e:
        st.error("❌ File is locked. Try these solutions:")
        st.markdown("""
        1. **Close Excel** or any program that might have the CSV open
        2. **Restart your computer** to release all file locks
        3. **Use Demo Data** for now and download Kaggle data manually later
        """)
        return None
    except Exception as e:
        st.error(f"Error loading Kaggle data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def load_uploaded_file(uploaded_file, max_rows=None):
    """Load user-uploaded CSV file with progress tracking."""
    try:
        progress = st.progress(0, text="📄 Reading uploaded file...")
        
        # Get file size
        file_size = uploaded_file.size
        st.info(f"📁 File: {uploaded_file.name} ({file_size / (1024*1024):.1f} MB)")
        
        progress.progress(20, text="📊 Parsing CSV data...")
        
        # Read with chunking for large files
        if max_rows:
            df = pd.read_csv(uploaded_file, nrows=max_rows, low_memory=False)
        else:
            # For very large files, sample
            if file_size > 500 * 1024 * 1024:  # > 500MB
                st.warning("⚠️ Large file detected. Loading first 200,000 rows for performance.")
                df = pd.read_csv(uploaded_file, nrows=200000, low_memory=False)
            else:
                df = pd.read_csv(uploaded_file, low_memory=False)
        
        progress.progress(60, text=f"✅ Loaded {len(df):,} rows")
        
        # Check columns
        st.write(f"**Columns found:** {', '.join(df.columns[:10].tolist())}{'...' if len(df.columns) > 10 else ''}")
        
        progress.progress(80, text="🎯 Calculating risk scores...")
        
        # Generate PD if not present
        if 'pd' not in df.columns:
            if 'grade' in df.columns:
                grade_pd = {'A': 0.05, 'B': 0.10, 'C': 0.18, 'D': 0.25, 'E': 0.35, 'F': 0.45, 'G': 0.55}
                df['pd'] = df['grade'].map(grade_pd).fillna(0.2) + np.random.normal(0, 0.02, len(df))
                df['pd'] = df['pd'].clip(0.01, 0.95)
            elif 'loan_status' in df.columns:
                df['pd'] = df['loan_status'].apply(lambda x: 0.75 if 'Charged' in str(x) or 'Default' in str(x) else 0.12)
            else:
                df['pd'] = np.random.uniform(0.08, 0.35, len(df))
        
        if 'loan_amnt' not in df.columns:
            if 'funded_amnt' in df.columns:
                df['loan_amnt'] = df['funded_amnt']
            else:
                df['loan_amnt'] = 15000
        
        progress.progress(100, text="✅ File loaded successfully!")
        st.success(f"✅ Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        return df
        
    except pd.errors.ParserError as e:
        st.error(f"❌ CSV parsing error: {e}")
        st.info("💡 Tip: Make sure the file is a valid CSV with proper formatting.")
        return None
    except MemoryError:
        st.error("❌ File too large for memory. Try uploading a smaller file or use the row limit option.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def calculate_portfolio_metrics(df):
    """Calculate portfolio metrics from dataframe."""
    lgd = 0.40
    pd_values = df['pd'].values if 'pd' in df.columns else np.ones(len(df)) * 0.15
    loan_amounts = df['loan_amnt'].values if 'loan_amnt' in df.columns else np.ones(len(df)) * 15000
    expected_loss = pd_values * loan_amounts * lgd
    
    high_risk = pd_values >= 0.5
    medium_risk = (pd_values >= 0.2) & (pd_values < 0.5)
    low_risk = pd_values < 0.2
    
    return {
        'total_loans': len(df),
        'total_exposure': loan_amounts.sum(),
        'average_loan_amount': loan_amounts.mean(),
        'total_expected_loss': expected_loss.sum(),
        'expected_loss_rate': expected_loss.sum() / loan_amounts.sum() if loan_amounts.sum() > 0 else 0,
        'average_pd': pd_values.mean(),
        'weighted_average_pd': (pd_values * loan_amounts).sum() / loan_amounts.sum() if loan_amounts.sum() > 0 else 0,
        'pd_std': pd_values.std(),
        'high_risk_count': int(high_risk.sum()),
        'high_risk_exposure': loan_amounts[high_risk].sum(),
        'high_risk_exposure_pct': loan_amounts[high_risk].sum() / loan_amounts.sum() if loan_amounts.sum() > 0 else 0,
        'medium_risk_count': int(medium_risk.sum()),
        'medium_risk_exposure': loan_amounts[medium_risk].sum(),
        'medium_risk_exposure_pct': loan_amounts[medium_risk].sum() / loan_amounts.sum() if loan_amounts.sum() > 0 else 0,
        'low_risk_count': int(low_risk.sum()),
        'low_risk_exposure': loan_amounts[low_risk].sum(),
        'low_risk_exposure_pct': loan_amounts[low_risk].sum() / loan_amounts.sum() if loan_amounts.sum() > 0 else 0,
        'lgd_assumption': lgd
    }


def main():
    """Main application."""
    initialize_session_state()
    
    # Sidebar
    st.sidebar.markdown("# 🏦 Credit Risk Platform")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Executive Dashboard", "🎯 Individual Risk", "📊 Portfolio Analysis", 
         "🔥 Risk Segmentation", "🧠 Model Insights", "⚙️ Data & Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Load Data")
    
    # All data loading options
    data_option = st.sidebar.radio(
        "Data Source:",
        ["🎲 Demo Data", "🌐 Kaggle Lending Club", "📤 Upload CSV", "📂 Data Folder"]
    )
    
    if data_option == "🎲 Demo Data":
        if st.sidebar.button("📥 Load Demo Data", use_container_width=True):
            with st.spinner("Generating demo data..."):
                st.session_state.portfolio_df = load_demo_data()
                st.session_state.portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio_df)
                st.session_state.data_loaded = True
                st.session_state.data_source = "Demo (Synthetic)"
            st.sidebar.success("Demo data loaded!")
    
    elif data_option == "🌐 Kaggle Lending Club":
        n_rows = st.sidebar.slider("Rows to load:", 10000, 100000, 50000, 10000)
        if st.sidebar.button("📥 Download from Kaggle", use_container_width=True):
            with st.spinner("Downloading Lending Club data..."):
                df = load_kaggle_data(n_rows)
                if df is not None:
                    st.session_state.portfolio_df = df
                    st.session_state.portfolio_metrics = calculate_portfolio_metrics(df)
                    st.session_state.data_loaded = True
                    st.session_state.data_source = "Kaggle Lending Club"
                    st.sidebar.success(f"✅ {len(df):,} loans loaded!")
    
    elif data_option == "📤 Upload CSV":
        st.sidebar.caption("⚠️ For files >100MB, use Data Folder option")
        max_rows = st.sidebar.number_input("Max rows (0=all):", 0, 500000, 50000, 10000)
        uploaded = st.sidebar.file_uploader("Upload CSV:", type=['csv'], key="csv_upload")
        
        if uploaded:
            st.sidebar.info(f"📁 {uploaded.name} ({uploaded.size // (1024*1024)}MB)")
            if st.sidebar.button("📤 Process File", use_container_width=True):
                try:
                    with st.spinner(f"Processing {uploaded.name}..."):
                        # Save to data/input first to avoid processing issues
                        data_folder = project_root / "data" / "input"
                        data_folder.mkdir(parents=True, exist_ok=True)
                        save_path = data_folder / uploaded.name
                        
                        # Write uploaded content
                        with open(save_path, 'wb') as f:
                            f.write(uploaded.getbuffer())
                        
                        st.info(f"✅ Saved to {save_path}")
                        
                        # Now read it
                        df = pd.read_csv(save_path, nrows=max_rows if max_rows > 0 else None, low_memory=False)
                        st.success(f"✅ Loaded {len(df):,} rows")
                        
                        # Generate risk scores
                        if 'pd' not in df.columns:
                            if 'grade' in df.columns:
                                grade_pd = {'A': 0.05, 'B': 0.10, 'C': 0.18, 'D': 0.25, 'E': 0.35, 'F': 0.45, 'G': 0.55}
                                df['pd'] = df['grade'].map(grade_pd).fillna(0.2) + np.random.normal(0, 0.02, len(df))
                                df['pd'] = df['pd'].clip(0.01, 0.95)
                            elif 'loan_status' in df.columns:
                                df['pd'] = df['loan_status'].apply(lambda x: 0.75 if 'Charged' in str(x) else 0.12)
                            else:
                                df['pd'] = np.random.uniform(0.08, 0.35, len(df))
                        
                        if 'loan_amnt' not in df.columns:
                            df['loan_amnt'] = df.get('funded_amnt', pd.Series([15000] * len(df)))
                        
                        st.session_state.portfolio_df = df
                        st.session_state.portfolio_metrics = calculate_portfolio_metrics(df)
                        st.session_state.data_loaded = True
                        st.session_state.data_source = f"Upload: {uploaded.name}"
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("💡 **Large file?** Use 'Data Folder' option instead.")
    
    elif data_option == "📂 Data Folder":
        st.sidebar.markdown("**No upload needed!**")
        st.sidebar.info("📁 Place your CSV in:\n`data/input/`")
        
        # Scan data folder
        data_folder = project_root / "data" / "input"
        data_folder.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(data_folder.glob("*.csv"))
        
        if csv_files:
            st.sidebar.success(f"✅ Found {len(csv_files)} file(s)")
            selected_file = st.sidebar.selectbox(
                "Select file:", 
                csv_files, 
                format_func=lambda x: f"{x.name} ({x.stat().st_size // (1024*1024)}MB)"
            )
            max_rows = st.sidebar.number_input("Max rows (0=all):", 0, 500000, 100000, 10000)
            
            if st.sidebar.button("📂 Load Selected File", use_container_width=True):
                with st.spinner(f"Loading {selected_file.name}..."):
                    try:
                        import subprocess
                        import tempfile
                        
                        # Copy to temp to avoid any locks
                        temp_file = Path(tempfile.gettempdir()) / f"credit_risk_{selected_file.name}"
                        result = subprocess.run(
                            ['cmd', '/c', 'copy', '/Y', str(selected_file), str(temp_file)],
                            capture_output=True, timeout=120
                        )
                        
                        # Read from temp copy
                        df = pd.read_csv(temp_file, nrows=max_rows if max_rows > 0 else None, low_memory=False)
                        st.success(f"✅ Loaded {len(df):,} rows from {selected_file.name}")
                        
                        # Generate PD
                        if 'pd' not in df.columns:
                            if 'grade' in df.columns:
                                grade_pd = {'A': 0.05, 'B': 0.10, 'C': 0.18, 'D': 0.25, 'E': 0.35, 'F': 0.45, 'G': 0.55}
                                df['pd'] = df['grade'].map(grade_pd).fillna(0.2) + np.random.normal(0, 0.02, len(df))
                                df['pd'] = df['pd'].clip(0.01, 0.95)
                            elif 'loan_status' in df.columns:
                                df['pd'] = df['loan_status'].apply(lambda x: 0.75 if 'Charged' in str(x) else 0.12)
                            else:
                                df['pd'] = np.random.uniform(0.08, 0.35, len(df))
                        if 'loan_amnt' not in df.columns:
                            df['loan_amnt'] = df.get('funded_amnt', pd.Series([15000] * len(df)))
                        
                        st.session_state.portfolio_df = df
                        st.session_state.portfolio_metrics = calculate_portfolio_metrics(df)
                        st.session_state.data_loaded = True
                        st.session_state.data_source = f"Folder: {selected_file.name}"
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.sidebar.warning("No CSV files found")
            st.sidebar.markdown(f"""
            **Steps:**
            1. Copy your CSV to: `{data_folder}`
            2. Refresh this page
            3. Select and load!
            """)
    
    # Show loaded status
    if st.session_state.data_loaded:
        st.sidebar.success(f"✅ {len(st.session_state.portfolio_df):,} loans loaded")
        st.sidebar.caption(f"Source: {st.session_state.data_source}")
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Tip:** For real analysis, use Kaggle data or upload your own CSV.")
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Enterprise Credit Risk Intelligence Platform</h1>
        <p>Advanced analytics for credit risk assessment and portfolio management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🏠 Executive Dashboard":
        from src.dashboard.pages.home import render_home_page
        render_home_page(
            st.session_state.portfolio_metrics,
            st.session_state.portfolio_df['pd'].values if st.session_state.data_loaded and 'pd' in st.session_state.portfolio_df.columns else None
        )
    
    elif page == "🎯 Individual Risk":
        from src.dashboard.pages.individual_risk import render_individual_risk_page
        render_individual_risk_page(st.session_state.model_trainer, st.session_state.shap_explainer, [])
    
    elif page == "📊 Portfolio Analysis":
        from src.dashboard.pages.portfolio_analysis import render_portfolio_analysis_page
        render_portfolio_analysis_page(st.session_state.portfolio_df)
    
    elif page == "🔥 Risk Segmentation":
        from src.dashboard.pages.risk_segmentation import render_risk_segmentation_page
        render_risk_segmentation_page(st.session_state.portfolio_df)
    
    elif page == "🧠 Model Insights":
        from src.dashboard.pages.model_insights import render_model_insights_page
        render_model_insights_page()
    
    elif page == "⚙️ Data & Settings":
        st.markdown("## ⚙️ Data & Settings")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Parameters")
            st.slider("Loss Given Default (LGD)", 0.1, 0.8, 0.4, 0.05)
            st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)
        
        with col2:
            st.markdown("### Data Info")
            if st.session_state.data_loaded:
                st.write(f"**Source:** {st.session_state.data_source}")
                st.write(f"**Records:** {len(st.session_state.portfolio_df):,}")
                st.write(f"**Columns:** {len(st.session_state.portfolio_df.columns)}")
            else:
                st.info("No data loaded yet.")


if __name__ == "__main__":
    main()
