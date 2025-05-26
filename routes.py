from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
import numpy as np
from bson.objectid import ObjectId
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
import requests
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import pandas as pd
from database import (
    them_tieu_chi, them_phuong_an, xoa_tieu_chi, xoa_phuong_an, xoa_ket_qua,
    lay_danh_sach_tieu_chi, lay_danh_sach_phuong_an, lay_danh_sach_ket_qua,
    luu_ma_tran_so_sanh, lay_ma_tran_so_sanh, luu_ket_qua, lay_ket_qua_theo_id, get_collections
)
from ahp_calculations import perform_ahp_steps, calculate_weights, normalize_matrix
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Blueprint
bp = Blueprint('routes', __name__)

# Chart Directory
CHART_DIR = os.path.join('static', 'charts')
if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# FMP API Key
FMP_API_KEY = 'wBgt7rXMIrwQb9PWOMUEQCYcZWjiWvZ9'

# Font Configuration
FONT_DIR = os.path.join(os.path.dirname(__file__), 'static', 'fonts')
FONT_PATH = os.path.join(FONT_DIR, 'DejaVuSans.ttf')

# Register font
def register_fonts():
    try:
        if os.path.exists(FONT_PATH):
            pdfmetrics.registerFont(TTFont('DejaVuSans', FONT_PATH))
            logger.info(f"Successfully registered font DejaVuSans from {FONT_PATH}")
        else:
            logger.warning(f"Font file {FONT_PATH} does not exist, using default font")
    except Exception as e:
        logger.error(f"Error registering font: {str(e)}")
register_fonts()

# Helper Functions
def normalize_tong_trong_so(value, tolerance=0.05):
    """Normalize and validate tong_trong_so as float."""
    try:
        if isinstance(value, (list, tuple)) and value:
            value = float(value[0])
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            raise ValueError("tong_trong_so không hợp lệ")
        if not (1.0 - tolerance <= value <= 1.0 + tolerance):
            raise ValueError(f"tong_trong_so {value} ngoài khoảng hợp lệ (0.95–1.05)")
        return value
    except (ValueError, TypeTypeError) as e:
        logger.error(f"Error normalizing tong_trong_so: {str(e)}")
        raise ValueError(f"Error normalizing tong_trong_so: {str(e)}")

def validate_matrix_doc(doc, matrix_type="criteria"):
    """Check if matrix document has all required keys."""
    required_keys = [
        'ma_tran', 'tong_cot', 'ma_tran_chuan_hoa', 'trong_so',
        'tong_trong_so', 'vector_nhat_quan', 'lambda_max', 'ci', 'cr'
    ]
    if not doc or not all(key in doc for key in required_keys):
        return False, f"Dữ liệu ma trận {matrix_type} không đầy đủ"
    try:
        doc['tong_trong_so'] = normalize_tong_trong_so(doc['tong_trong_so'])
    except ValueError as e:
        return False, f"Lỗi dữ liệu {matrix_type}: {str(e)}"
    return True, ""

def get_fmp_data(endpoint, params=None):
    """Fetch data from Financial Modeling Prep API."""
    base_url = "https://financialmodelingprep.com/api/v3"
    url = f"{base_url}/{endpoint}?apikey={FMP_API_KEY}"
    if params:
        url += "&" + "&".join([f"{k}={v}" for k, v in params.items()])
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching FMP data for {endpoint}: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {endpoint}: {str(e)}")
        return None

def get_alpha_vantage_economic(indicator):
    """Fetch economic data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function={indicator}&apikey=LJJ96MCARKRTXEVW"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                return data["data"]
        logger.error(f"Error fetching Alpha Vantage data for {indicator}: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {indicator}: {str(e)}")
        return None

def find_matrix_start(df, keyword, max_rows=100):
    """Find the starting row of a matrix based on a keyword."""
    for i in range(min(max_rows, df.shape[0])):
        for j in range(df.shape[1]):
            if pd.notna(df.iloc[i, j]) and keyword.lower() in str(df.iloc[i, j]).lower():
                logger.info(f"Found keyword '{keyword}' at row {i}, col {j}")
                return i
    logger.error(f"Keyword '{keyword}' not found in Excel")
    return None

def read_matrix_from_df(df, start_row, n):
    """Read a square matrix from the DataFrame."""
    try:
        matrix = df.iloc[start_row:start_row + n, 1:n+1].values
        matrix = np.array(matrix, dtype=float)
        if np.any(np.isnan(matrix)):
            raise ValueError("Matrix contains NaN values")
        logger.info(f"Read matrix at row {start_row}:\n{matrix}")
        return matrix
    except Exception as e:
        logger.error(f"Failed to read matrix at row {start_row}: {str(e)}")
        raise ValueError(f"Failed to read matrix: {str(e)}")

def read_tong_cot_from_df(df, row, n):
    """Read the 'Tổng cột' values."""
    try:
        tong_cot = df.iloc[row, 1:n+1].values
        tong_cot = np.array(tong_cot, dtype=float)
        if np.any(np.isnan(tong_cot)):
            raise ValueError("Tổng cột contains NaN values")
        logger.info(f"Read tổng cột at row {row}: {tong_cot}")
        return tong_cot.tolist()
    except Exception as e:
        logger.error(f"Failed to read tổng cột at row {row}: {str(e)}")
        raise ValueError(f"Failed to read tổng cột: {str(e)}")

def read_normalized_matrix_from_df(df, start_row, n):
    """Read the normalized matrix."""
    try:
        normalized = df.iloc[start_row:start_row + n, 1:n+1].values
        normalized = np.array(normalized, dtype=float)
        if np.any(np.isnan(normalized)):
            raise ValueError("Normalized matrix contains NaN values")
        logger.info(f"Read normalized matrix at row {start_row}:\n{normalized}")
        return normalized.tolist()
    except Exception as e:
        logger.error(f"Failed to read normalized matrix at row {start_row}: {str(e)}")
        raise ValueError(f"Failed to read normalized matrix: {str(e)}")

def read_trong_so_from_df(df, start_row, n, col_idx):
    """Read the 'Trọng số' values from a specific column."""
    try:
        trong_so = df.iloc[start_row:start_row + n, col_idx].values
        trong_so = [float(x) if pd.notna(x) else 0.0 for x in trong_so]
        if any(np.isnan(trong_so)):
            raise ValueError("Trọng số contains invalid values")
        logger.info(f"Read trọng số at row {start_row}, col {col_idx}: {trong_so}")
        return trong_so
    except Exception as e:
        logger.error(f"Failed to read trọng số at row {start_row}, col {col_idx}: {str(e)}")
        raise ValueError(f"Failed to read trọng số: {str(e)}")

def read_vector_nhat_quan_from_df(df, start_row, n, col_idx):
    """Read the consistency vector."""
    try:
        vector = df.iloc[start_row:start_row + n, col_idx].values
        vector = [float(x) if pd.notna(x) and isinstance(x, (int, float, str)) and str(x).replace('.', '', 1).isdigit() else 0.0 for x in vector]
        logger.info(f"Read vector nhất quán at row {start_row}, col {col_idx}: {vector}")
        return vector
    except Exception as e:
        logger.error(f"Failed to read vector nhất quán at row {start_row}, col {col_idx}: {str(e)}")
        return [0.0] * n

# Routes
@bp.route('/', methods=['GET', 'POST'])
def index():
    """Main page with criteria, alternatives, and results."""
    if request.method == 'POST':
        try:
            if 'add_criteria' in request.form:
                criteria = request.form['criteria'].strip()
                if criteria:
                    them_tieu_chi(criteria)
                    flash("Thêm tiêu chí thành công!", "success")
                else:
                    flash("Tiêu chí không được để trống!", "danger")
            elif 'add_alternative' in request.form:
                alternative = request.form['alternative'].strip()
                if alternative:
                    them_phuong_an(alternative)
                    flash("Thêm phương án thành công!", "success")
                else:
                    flash("Phương án không được để trống!", "danger")
            elif 'delete_criteria' in request.form:
                crit_id = request.form['crit_id']
                crit_name = request.form['crit_name']
                xoa_tieu_chi(crit_id, crit_name)
                flash("Xóa tiêu chí thành công! Vui lòng cập nhật ma trận liên quan.", "success")
            elif 'delete_alternative' in request.form:
                alt_id = request.form['alt_id']
                xoa_phuong_an(alt_id)
                flash("Xóa phương án thành công! Vui lòng cập nhật ma trận liên quan.", "success")
            elif 'delete_result' in request.form:
                result_id = request.form['result_id']
                xoa_ket_qua(result_id)
                flash("Xóa kết quả thành công!", "success")
        except Exception as e:
            logger.error(f"Error processing POST request: {str(e)}")
            flash(f"Lỗi khi xử lý yêu cầu: {str(e)}", "danger")
    
    try:
        criteria = lay_danh_sach_tieu_chi()
        alternatives = lay_danh_sach_phuong_an()
        results = lay_danh_sach_ket_qua()
        logger.info(f"Loaded index with {len(criteria)} criteria, {len(alternatives)} alternatives, {len(results)} results")
        return render_template('index.html', criteria=criteria, alternatives=alternatives, results=results)
    except Exception as e:
        logger.error(f"Error loading index page: {str(e)}")
        flash(f"Lỗi khi tải trang chủ: {str(e)}", "danger")
        return render_template('index.html', criteria=[], alternatives=[], results=[])

@bp.route('/import_excel', methods=['POST'])
def import_excel():
    """Import AHP data from an Excel file."""
    logger.info("Received request for /import_excel")
    logger.info(f"Form data: {request.form}")
    logger.info(f"Files: {request.files}")
    
    if 'import_excel' not in request.form or 'excel_file' not in request.files:
        logger.error("Missing import_excel key or excel_file in request")
        flash("Không tìm thấy tệp Excel hoặc yêu cầu không hợp lệ!", "danger")
        return redirect(url_for('routes.index'))

    file = request.files['excel_file']
    if file.filename == '':
        logger.error("No file selected")
        flash("Vui lòng chọn một tệp Excel!", "danger")
        return redirect(url_for('routes.index'))

    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        logger.error("Invalid file format")
        flash("Tệp phải có định dạng .xlsx hoặc .xls!", "danger")
        return redirect(url_for('routes.index'))

    try:
        logger.info("Reading Excel file")
        xls = pd.ExcelFile(file)
        logger.info(f"Sheet names: {xls.sheet_names}")
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"DataFrame head (first 20 rows):\n{df.iloc[:20].to_string()}")

        # Get collections
        collections = get_collections()
        
        # Clear existing data
        logger.info("Clearing existing criteria, alternatives, matrices, and results")
        collections['criteria'].delete_many({})
        collections['alternatives'].delete_many({})
        collections['pairwise_comparisons'].delete_many({})
        collections['results'].delete_many({})

        # Define criteria and alternatives
        criteria_names = [
            'Hệ số kỳ vọng', 'Rủi ro đầu tư',
            'Tình trạng khoản đầu tư', 'Mức độ ảnh hưởng'
        ]
        alternative_names = [
            'Cổ phiếu lướt sóng', 'Cổ phiếu blue-chip',
            'Trái phiếu', 'ETF', 'Vàng'
        ]

        # Add criteria
        criteria_ids = []
        for crit in criteria_names:
            logger.info(f"Adding criterion: {crit}")
            result = them_tieu_chi(crit)
            if not result:
                logger.error(f"Failed to add criterion: {crit}")
                flash(f"Không thể thêm tiêu chí {crit}!", "danger")
                return redirect(url_for('routes.index'))
            criteria_ids.append(result.inserted_id)
        
        # Add alternatives
        alternative_ids = []
        for alt in alternative_names:
            logger.info(f"Adding alternative: {alt}")
            result = them_phuong_an(alt)
            if not result:
                logger.error(f"Failed to add alternative: {alt}")
                flash(f"Không thể thêm phương án {alt}!", "danger")
                return redirect(url_for('routes.index'))
            alternative_ids.append(result.inserted_id)

        # Verify saved data
        saved_criteria = lay_danh_sach_tieu_chi()
        saved_alternatives = lay_danh_sach_phuong_an()
        logger.info(f"Saved criteria: {[c['ten'] for c in saved_criteria]}")
        logger.info(f"Saved alternatives: {[a['ten'] for a in saved_alternatives]}")
        if len(saved_criteria) != len(criteria_names) or len(saved_alternatives) != len(alternative_names):
            logger.error("Mismatch in saved criteria or alternatives")
            flash("Lỗi: Không lưu được đầy đủ tiêu chí hoặc phương án!", "danger")
            return redirect(url_for('routes.index'))

        # Parse criteria matrix
        n_criteria = len(criteria_names)
        criteria_start = find_matrix_start(df, "Hệ số kỳ vọng")
        if criteria_start is None:
            logger.error("Cannot find criteria matrix")
            flash("Không tìm thấy ma trận tiêu chí trong Excel!", "danger")
            return redirect(url_for('routes.index'))

        try:
            # Adjust for header row
            matrix_start = criteria_start + 1
            criteria_matrix = read_matrix_from_df(df, matrix_start, n_criteria)
            logger.info(f"Criteria matrix:\n{criteria_matrix}")
            criteria_tong_cot = read_tong_cot_from_df(df, matrix_start + n_criteria, n_criteria)
            criteria_normalized = read_normalized_matrix_from_df(df, matrix_start + n_criteria + 1, n_criteria)
            criteria_trong_so = read_trong_so_from_df(df, matrix_start + n_criteria + 1, n_criteria, col_idx=5)  # Column 6
            criteria_vector_nhat_quan = read_vector_nhat_quan_from_df(df, matrix_start + n_criteria + 1, n_criteria, col_idx=6)  # Column 7
            
            # Validate trong_so
            criteria_tong_trong_so = sum(criteria_trong_so)
            criteria_tong_trong_so = normalize_tong_trong_so(criteria_tong_trong_so)
            
            # AHP parameters (dynamic search)
            param_start = matrix_start + n_criteria + n_criteria + 1
            criteria_lambda_max = 0.0
            criteria_ci = 0.0
            criteria_cr = 0.0
            for i in range(param_start, min(param_start + 10, df.shape[0])):
                if pd.notna(df.iloc[i, 0]):
                    cell = str(df.iloc[i, 0]).lower()
                    if 'λ_max' in cell or 'lambda' in cell:
                        criteria_lambda_max = float(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else 0.0
                    elif 'ci' in cell:
                        criteria_ci = float(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else 0.0
                    elif 'cr' in cell:
                        criteria_cr = float(df.iloc[i, 1]) if pd.notna(df.iloc[i, 1]) else 0.0
            logger.info(f"Criteria AHP params: lambda_max={criteria_lambda_max}, ci={criteria_ci}, cr={criteria_cr}, tong_trong_so={criteria_tong_trong_so}")
        except Exception as e:
            logger.error(f"Error parsing criteria matrix: {str(e)}")
            flash(f"Lỗi khi đọc ma trận tiêu chí: {str(e)}", "danger")
            return redirect(url_for('routes.index'))

        criteria_matrix_doc = {
            "ma_tran": criteria_matrix.tolist(),
            "tong_cot": criteria_tong_cot,
            "ma_tran_chuan_hoa": criteria_normalized,
            "trong_so": criteria_trong_so,
            "tong_trong_so": criteria_tong_trong_so,
            "vector_nhat_quan": criteria_vector_nhat_quan,
            "lambda_max": criteria_lambda_max,
            "ci": criteria_ci,
            "cr": criteria_cr,
            "thoi_gian": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        logger.info(f"Saving criteria matrix: {criteria_matrix_doc}")
        result = luu_ma_tran_so_sanh("criteria", None, criteria_matrix_doc)
        if not result:
            logger.error("Failed to save criteria matrix")
            flash("Không thể lưu ma trận tiêu chí!", "danger")
        else:
            logger.info("Successfully saved criteria matrix")
            flash("Lưu ma trận tiêu chí thành công!", "success")

        # Parse alternative matrices
        alt_matrices = []
        alt_weights_per_criterion = []
        n_alternatives = len(alternative_names)
        alt_start_rows = []
        expected_starts = [11, 30, 49, 68]  # Based on Excel structure

        # Find start rows for alternative matrices
        for i, crit in enumerate(criteria_names):
            start_row = find_matrix_start(df, crit, max_rows=df.shape[0])
            if start_row is None or abs(start_row - expected_starts[i]) > 5:
                logger.error(f"Cannot find or incorrect alternative matrix for criterion: {crit}")
                flash(f"Không tìm thấy ma trận phương án cho tiêu chí {crit}!", "danger")
                return redirect(url_for('routes.index'))
            alt_start_rows.append(start_row)

        for i, crit_name in enumerate(criteria_names):
            start_row = alt_start_rows[i]
            logger.info(f"Reading alternative matrix for {crit_name} at row {start_row}")
            try:
                # Adjust for header row
                matrix_start = start_row + 1
                alt_matrix = read_matrix_from_df(df, matrix_start, n_alternatives)
                alt_tong_cot = read_tong_cot_from_df(df, matrix_start + n_alternatives, n_alternatives)
                alt_normalized = read_normalized_matrix_from_df(df, matrix_start + n_alternatives + 1, n_alternatives)
                alt_trong_so = read_trong_so_from_df(df, matrix_start + n_alternatives + 1, n_alternatives, col_idx=6)  # Column 7
                alt_vector_nhat_quan = read_vector_nhat_quan_from_df(df, matrix_start + n_alternatives + 1, n_alternatives, col_idx=7)  # Column 8
                
                # Validate trong_so
                alt_tong_trong_so = sum(alt_trong_so)
                alt_tong_trong_so = normalize_tong_trong_so(alt_tong_trong_so)
                
                # AHP parameters
                param_start = matrix_start + n_alternatives + n_alternatives + 5
                alt_lambda_max = float(df.iloc[param_start + 11, 7]) if pd.notna(df.iloc[param_start + 11, 7]) else 0.0
                alt_ci = float(df.iloc[param_start + 12, 8]) if pd.notna(df.iloc[param_start + 12, 8]) else 0.0
                alt_cr = float(df.iloc[param_start + 13, 9]) if pd.notna(df.iloc[param_start + 13, 9]) else 0.0
                logger.info(f"Alternative matrix for {crit_name}:\n{alt_matrix}")
                logger.info(f"Alternative AHP params: lambda_max={alt_lambda_max}, ci={alt_ci}, cr={alt_cr}, tong_trong_so={alt_tong_trong_so}")
            except Exception as e:
                logger.error(f"Error parsing alternative matrix for {crit_name}: {str(e)}")
                flash(f"Lỗi khi đọc ma trận phương án cho {crit_name}: {str(e)}", "danger")
                return redirect(url_for('routes.index'))

            alt_matrix_doc = {
                "ma_tran": alt_matrix.tolist(),
                "tong_cot": alt_tong_cot,
                "ma_tran_chuan_hoa": alt_normalized,
                "trong_so": alt_trong_so,
                "tong_trong_so": alt_tong_trong_so,
                "vector_nhat_quan": alt_vector_nhat_quan,
                "lambda_max": alt_lambda_max,
                "ci": alt_ci,
                "cr": alt_cr,
                "thoi_gian": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            logger.info(f"Saving alternative matrix for {crit_name}: {alt_matrix_doc}")
            result = luu_ma_tran_so_sanh("phuong_an", crit_name, alt_matrix_doc)
            if not result:
                logger.error(f"Failed to save alternative matrix for {crit_name}")
                flash(f"Không thể lưu ma trận phương án cho {crit_name}!", "danger")
            else:
                logger.info(f"Successfully saved alternative matrix for {crit_name}")
                flash(f"Lưu ma trận phương án cho {crit_name} thành công!", "success")
            alt_matrices.append(alt_matrix_doc)
            alt_weights_per_criterion.append(alt_trong_so)

        # Calculate final scores
        try:
            criteria_weights = np.array(criteria_trong_so)
            final_scores = np.zeros(n_alternatives)
            for i in range(n_criteria):
                final_scores += criteria_weights[i] * np.array(alt_weights_per_criterion[i])
            logger.info(f"Final scores: {final_scores}")

            ranking = [
                {"name": alt, "score": float(score)}
                for alt, score in zip(alternative_names, final_scores)
            ]
            ranking.sort(key=lambda x: x['score'], reverse=True)
            logger.info(f"Ranking: {ranking}")
        except Exception as e:
            logger.error(f"Error calculating final scores: {str(e)}")
            flash(f"Lỗi khi tính điểm số cuối cùng: {str(e)}", "danger")
            return redirect(url_for('routes.index'))

        # Create ranking chart
        try:
            plt.figure(figsize=(10, 6))
            plt.rcParams['font.family'] = 'DejaVuSans' if os.path.exists(FONT_PATH) else 'sans-serif'
            plt.bar([item['name'] for item in ranking], [item['score'] for item in ranking], color='skyblue')
            plt.xlabel('Phương án')
            plt.ylabel('Điểm số')
            plt.title('Xếp hạng đầu tư')
            plt.xticks(rotation=45)
            chart_filename = f"ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = os.path.join(CHART_DIR, chart_filename)
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved ranking chart: {chart_path}")
        except Exception as e:
            logger.error(f"Error creating ranking chart: {str(e)}")
            flash(f"Lỗi khi tạo biểu đồ xếp hạng: {str(e)}", "danger")
            return redirect(url_for('routes.index'))

        # Save result
        result_doc = {
            "thoi_gian": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "criteria_matrix": criteria_matrix_doc,
            "criteria_names": criteria_names,
            "alternative_names": alternative_names,
            "alt_matrices": alt_matrices,
            "alt_weights_per_criterion": [weights.tolist() for weights in alt_weights_per_criterion],
            "ranking": ranking,
            "chart": os.path.join('charts', chart_filename).replace('\\', '/')
        }
        logger.info("Saving AHP result")
        result = luu_ket_qua(result_doc)
        if not result:
            logger.error("Failed to save AHP result")
            flash("Không thể lưu kết quả AHP!", "danger")
            return redirect(url_for('routes.index'))

        flash("Nhập Excel thành công! Tiêu chí, phương án và kết quả đã được lưu.", "success")
        logger.info("Excel import completed successfully")
    except pd.errors.EmptyDataError:
        logger.error("Excel file is empty or invalid")
        flash("Tệp Excel trống hoặc không hợp lệ!", "danger")
    except Exception as e:
        logger.error(f"Error importing Excel: {str(e)}")
        flash(f"Lỗi khi nhập Excel: {str(e)}", "danger")
    
    return redirect(url_for('routes.index'))

@bp.route('/export_pdf/<result_id>')
def export_pdf(result_id):
    """Export AHP result as PDF."""
    try:
        result = lay_ket_qua_theo_id(result_id)
        if not result:
            flash("Kết quả không tìm thấy!", "danger")
            return redirect(url_for('routes.index'))

        valid, error = validate_matrix_doc(result.get('criteria_matrix', {}), "criteria")
        if not valid:
            flash(error, "danger")
            return redirect(url_for('routes.index'))
        for i, crit in enumerate(result.get('criteria_names', [])):
            valid, error = validate_matrix_doc(result['alt_matrices'][i], f"phương án {crit}")
            if not valid:
                flash(error, "danger")
                return redirect(url_for('routes.index'))

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name='Title',
            fontName='DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica',
            fontSize=16,
            leading=20,
            alignment=1,
            spaceAfter=20
        )
        heading_style = ParagraphStyle(
            name='Heading',
            fontName='DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica',
            fontSize=12,
            leading=15,
            spaceAfter=10
        )
        normal_style = ParagraphStyle(
            name='Normal',
            fontName='DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica',
            fontSize=10,
            leading=12
        )

        elements.append(Paragraph("Báo Cáo Kết Quả AHP", title_style))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Thông tin chung", heading_style))
        elements.append(Paragraph(f"<b>Thời gian:</b> {result.get('thoi_gian', 'N/A')}", normal_style))
        elements.append(Spacer(1, 12))

        def create_bar_chart(labels, data, title, filename, stacked=False, datasets=None):
            plt.figure(figsize=(6, 4), dpi=150)
            plt.rcParams['font.family'] = 'DejaVuSans' if os.path.exists(FONT_PATH) else 'sans-serif'
            colors_list = ['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'lavender']
            if stacked and datasets:
                bottom = np.zeros(len(labels))
                for idx, dataset in enumerate(datasets):
                    plt.bar(labels, dataset['data'], bottom=bottom, label=dataset['label'], color=colors_list[idx % len(colors_list)])
                    bottom += np.array(dataset['data'])
                plt.legend(fontsize=8)
            else:
                plt.bar(labels, data, color='skyblue', edgecolor='blue')
            plt.title(title, fontsize=10)
            plt.xlabel('Phương án/Tiêu chí', fontsize=8)
            plt.ylabel('Trọng số/Điểm số', fontsize=8)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            chart_path = os.path.join(CHART_DIR, filename)
            os.makedirs(CHART_DIR, exist_ok=True)
            plt.savefig(chart_path, format='png', bbox_inches='tight')
            plt.close()
            return chart_path

        chart_files = []

        elements.append(Paragraph("Bước 1: Phân tích ma trận tiêu chí", heading_style))
        if result.get('criteria_names') and result.get('criteria_matrix'):
            data = [[''] + result['criteria_names']]
            for i, name in enumerate(result['criteria_names']):
                row = [name] + [f"{x:.4f}" for x in result['criteria_matrix']['ma_tran'][i]]
                data.append(row)
            data.append(['Tổng cột'] + [f"{x:.4f}" for x in result['criteria_matrix']['tong_cot']])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(Paragraph("Ma trận so sánh cặp ban đầu", normal_style))
            elements.append(table)
            elements.append(Spacer(1, 12))

            data = [[''] + result['criteria_names']]
            for i, name in enumerate(result['criteria_names']):
                row = [name] + [f"{x:.4f}" for x in result['criteria_matrix']['ma_tran_chuan_hoa'][i]]
                data.append(row)
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(Paragraph("Ma trận chuẩn hóa", normal_style))
            elements.append(table)
            elements.append(Spacer(1, 12))

            data = [['Tiêu chí', 'Trọng số']]
            for i, name in enumerate(result['criteria_names']):
                data.append([name, f"{result['criteria_matrix']['trong_so'][i]:.4f}"])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(Paragraph("Trọng số tiêu chí", normal_style))
            elements.append(table)
            elements.append(Spacer(1, 12))

            chart_filename = f"criteria_weights_{result_id}.png"
            chart_path = create_bar_chart(result['criteria_names'], result['criteria_matrix']['trong_so'], "Trọng số tiêu chí", chart_filename)
            elements.append(Image(chart_path, width=350, height=175))
            chart_files.append(chart_path)
            elements.append(Spacer(1, 12))

            elements.append(Paragraph("Kiểm tra tính nhất quán", normal_style))
            tong_trong_so = result['criteria_matrix'].get('tong_trong_so', 'N/A')
            tong_trong_so_str = f"{tong_trong_so:.4f}" if isinstance(tong_trong_so, (int, float)) else 'N/A'
            elements.append(Paragraph(f"Tổng trọng số: {tong_trong_so_str}", normal_style))
            vector_nhat_quan = result['criteria_matrix'].get('vector_nhat_quan', [])
            elements.append(Paragraph(f"Vector nhất quán: {[round(x, 4) for x in vector_nhat_quan] if vector_nhat_quan else 'N/A'}", normal_style))
            elements.append(Paragraph(f"Lambda Max: {result['criteria_matrix'].get('lambda_max', 'N/A'):.4f}" if result['criteria_matrix'].get('lambda_max') is not None else "Lambda Max: N/A", normal_style))
            elements.append(Paragraph(f"Chỉ số nhất quán (CI): {result['criteria_matrix'].get('ci', 'N/A'):.4f}" if result['criteria_matrix'].get('ci') is not None else "Chỉ số nhất quán (CI): N/A", normal_style))
            elements.append(Paragraph(f"Tỷ số nhất quán (CR): {result['criteria_matrix'].get('cr', 'N/A'):.4f}" if result['criteria_matrix'].get('cr') is not None else "Tỷ số nhất quán (CR): N/A", normal_style))
            cr_message = "CR < 0.1: Ma trận nhất quán." if result['criteria_matrix'].get('cr', 1) < 0.1 else "CR >= 0.1: Ma trận không nhất quán."
            elements.append(Paragraph(cr_message, normal_style))
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Không có dữ liệu ma trận tiêu chí.", normal_style))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("Bước 2: Phân tích ma trận phương án", heading_style))
        if result.get('criteria_names') and result.get('alternative_names') and result.get('alt_matrices'):
            for i, crit_name in enumerate(result['criteria_names']):
                elements.append(Paragraph(f"Ma trận phương án cho {crit_name}", heading_style))

                data = [[''] + result['alternative_names']]
                for j, name in enumerate(result['alternative_names']):
                    row = [name] + [f"{x:.4f}" for x in result['alt_matrices'][i]['ma_tran'][j]]
                    data.append(row)
                data.append(['Tổng cột'] + [f"{x:.4f}" for x in result['alt_matrices'][i]['tong_cot']])
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(Paragraph("Ma trận so sánh cặp ban đầu", normal_style))
                elements.append(table)
                elements.append(Spacer(1, 12))

                data = [[''] + result['alternative_names']]
                for j, name in enumerate(result['alternative_names']):
                    row = [name] + [f"{x:.4f}" for x in result['alt_matrices'][i]['ma_tran_chuan_hoa'][j]]
                    data.append(row)
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(Paragraph("Ma trận chuẩn hóa", normal_style))
                elements.append(table)
                elements.append(Spacer(1, 12))

                data = [['Phương án', 'Trọng số']]
                for j, name in enumerate(result['alternative_names']):
                    data.append([name, f"{result['alt_weights_per_criterion'][i][j]:.4f}"])
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(Paragraph("Trọng số phương án", normal_style))
                elements.append(table)
                elements.append(Spacer(1, 12))

                chart_filename = f"alt_weights_{result_id}_{i}.png"
                chart_path = create_bar_chart(result['alternative_names'], result['alt_weights_per_criterion'][i], f"Trọng số phương án cho {crit_name}", chart_filename)
                elements.append(Image(chart_path, width=350, height=175))
                chart_files.append(chart_path)
                elements.append(Spacer(1, 12))

                elements.append(Paragraph("Kiểm tra tính nhất quán", normal_style))
                tong_trong_so = result['alt_matrices'][i].get('tong_trong_so', 'N/A')
                tong_trong_so_str = f"{tong_trong_so:.4f}" if isinstance(tong_trong_so, (int, float)) else 'N/A'
                elements.append(Paragraph(f"Tổng trọng số: {tong_trong_so_str}", normal_style))
                vector_nhat_quan = result['alt_matrices'][i].get('vector_nhat_quan', [])
                elements.append(Paragraph(f"Vector nhất quán: {[round(x, 4) for x in vector_nhat_quan] if vector_nhat_quan else 'N/A'}", normal_style))
                elements.append(Paragraph(f"Lambda Max: {result['alt_matrices'][i].get('lambda_max', 'N/A'):.4f}" if result['alt_matrices'][i].get('lambda_max') is not None else "Lambda Max: N/A", normal_style))
                elements.append(Paragraph(f"Chỉ số nhất quán (CI): {result['alt_matrices'][i].get('ci', 'N/A'):.4f}" if result['alt_matrices'][i].get('ci') is not None else "Chỉ số nhất quán (CI): N/A", normal_style))
                elements.append(Paragraph(f"Tỷ số nhất quán (CR): {result['alt_matrices'][i].get('cr', 'N/A'):.4f}" if result['alt_matrices'][i].get('cr') is not None else "Tỷ số nhất quán (CR): N/A", normal_style))
                cr_message = "CR < 0.1: Ma trận nhất quán." if result['alt_matrices'][i].get('cr', 1) < 0.1 else "CR >= 0.1: Ma trận không nhất quán."
                elements.append(Paragraph(cr_message, normal_style))
                elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Không có dữ liệu ma trận phương án.", normal_style))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("Góp phần của tiêu chí vào trọng số phương án", heading_style))
        if result.get('criteria_names') and result.get('alternative_names') and result.get('alt_weights_per_criterion'):
            datasets = [
                {
                    'label': crit_name,
                    'data': [result['alt_weights_per_criterion'][i][j] * result['criteria_matrix']['trong_so'][i] for j in range(len(result['alternative_names']))]
                }
                for i, crit_name in enumerate(result['criteria_names'])
            ]
            chart_filename = f"criteria_contribution_{result_id}.png"
            chart_path = create_bar_chart(
                result['alternative_names'],
                None,
                "Góp phần của tiêu chí vào trọng số phương án",
                chart_filename,
                stacked=True,
                datasets=datasets
            )
            elements.append(Image(chart_path, width=350, height=175))
            chart_files.append(chart_path)
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Không có dữ liệu để hiển thị biểu đồ góp phần.", normal_style))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("Góp phần của tiêu chí vào điểm số cuối cùng", heading_style))
        if result.get('ranking') and result.get('criteria_names') and result.get('alt_weights_per_criterion'):
            datasets = [
                {
                    'label': crit_name,
                    'data': [result['alt_weights_per_criterion'][i][j] * result['criteria_matrix']['trong_so'][i] for j in range(len(result['alternative_names']))]
                }
                for i, crit_name in enumerate(result['criteria_names'])
            ]
            chart_filename = f"final_score_contribution_{result_id}.png"
            chart_path = create_bar_chart(
                result['alternative_names'],
                None,
                "Góp phần của tiêu chí vào điểm số cuối cùng",
                chart_filename,
                stacked=True,
                datasets=datasets
            )
            elements.append(Image(chart_path, width=350, height=175))
            chart_files.append(chart_path)
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Không có dữ liệu để hiển thị biểu đồ góp phần.", normal_style))
            elements.append(Spacer(1, 12))

        elements.append(Paragraph("Bước 3: Xếp hạng cuối cùng", heading_style))
        if result.get('ranking'):
            data = [['Phương án', 'Điểm số', 'Xếp hạng']]
            for i, item in enumerate(result['ranking']):
                data.append([item['name'], f"{item['score']:.4f}", str(i + 1)])
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'DejaVuSans' if os.path.exists(FONT_PATH) else 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

            chart_filename = f"ranking_{result_id}.png"
            chart_path = create_bar_chart([item['name'] for item in result['ranking']], [item['score'] for item in result['ranking']], "Xếp hạng cuối cùng", chart_filename)
            elements.append(Image(chart_path, width=350, height=175))
            chart_files.append(chart_path)
            elements.append(Spacer(1, 12))
        else:
            elements.append(Paragraph("Không có dữ liệu xếp hạng.", normal_style))
            elements.append(Spacer(1, 12))

        doc.build(elements)
        buffer.seek(0)

        for chart_path in chart_files:
            if os.path.exists(chart_path):
                try:
                    os.remove(chart_path)
                    logger.info(f"Deleted chart file {chart_path}")
                except Exception as e:
                    logger.error(f"Error deleting chart file {chart_path}: {str(e)}")

        return send_file(buffer, as_attachment=True, download_name=f"ahp_report_{result_id}.pdf", mimetype='application/pdf')
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        flash(f"Lỗi khi tạo báo cáo PDF: {str(e)}", "danger")
        return redirect(url_for('routes.index'))

@bp.route('/api/stock_data/<symbol>', methods=['GET'])
def stock_data(symbol):
    """Fetch stock data from FMP."""
    data = get_fmp_data(f"quote/{symbol}")
    if data:
        return jsonify(data[0])
    return jsonify({"error": "Không thể lấy dữ liệu cổ phiếu"}), 500

@bp.route('/api/etf_data/<symbol>', methods=['GET'])
def etf_data(symbol):
    """Fetch ETF data from FMP."""
    data = get_fmp_data(f"etf/profile/{symbol}")
    if data:
        return jsonify(data[0])
    return jsonify({"error": "Không thể lấy dữ liệu ETF"}), 500

@bp.route('/api/crypto_data/<symbol>', methods=['GET'])
def crypto_data(symbol):
    """Fetch crypto data from FMP."""
    data = get_fmp_data(f"quote/{symbol}-USD", {"type": "crypto"})
    if data:
        return jsonify(data[0])
    return jsonify({"error": "Không thể lấy dữ liệu tiền điện tử"}), 500

@bp.route('/api/forex_data/<pair>', methods=['GET'])
def forex_data(pair):
    """Fetch forex data from FMP."""
    data = get_fmp_data(f"fx/{pair}")
    if data:
        return jsonify(data[0])
    return jsonify({"error": "Không thể lấy dữ liệu ngoại hối"}), 500

@bp.route('/api/commodity_data/<symbol>', methods=['GET'])
def commodity_data(symbol):
    """Fetch commodity data from FMP."""
    data = get_fmp_data(f"quote/{symbol}")
    if data:
        return jsonify(data[0])
    return jsonify({"error": "Không thể lấy dữ liệu hàng hóa"}), 500

@bp.route('/api/economic_indicators', methods=['GET'])
def economic_indicators():
    """Fetch economic indicators from FMP."""
    data = get_fmp_data("economic")
    if data:
        return jsonify(data)
    return jsonify({"error": "Không thể lấy dữ liệu chỉ số kinh tế"}), 500

@bp.route('/api/finance_data', methods=['GET'])
def finance_data():
    """Fetch various financial data."""
    try:
        stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        stock_data = get_fmp_data(f"quote/{','.join(stock_symbols)}")
        stock_result = [
            {
                "symbol": stock['symbol'],
                "price": stock['price'],
                "change": stock['change'],
                "volume": stock['volume'],
                "updated_at": (
                    datetime.fromtimestamp(stock['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'timestamp' in stock
                    else stock.get('lastUpdated', 'N/A')
                )
            } for stock in stock_data
        ] if stock_data else []

        etf_symbols = ['SPY', 'QQQ']
        etf_data = get_fmp_data(f"quote/{','.join(etf_symbols)}")
        etf_result = [
            {
                "symbol": etf['symbol'],
                "price": etf['price'],
                "description": "N/A",
                "updated_at": (
                    datetime.fromtimestamp(etf['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'timestamp' in etf
                    else etf.get('lastUpdated', 'N/A')
                )
            } for etf in etf_data
        ] if etf_data else []

        crypto_symbols = ['BTC', 'ETH']
        crypto_data = get_fmp_data(f"quote/{','.join([s + '-USD' for s in crypto_symbols])}")
        crypto_result = [
            {
                "symbol": crypto['symbol'],
                "price": crypto['price'],
                "change": crypto['change'],
                "volume": crypto['volume'],
                "updated_at": (
                    datetime.fromtimestamp(crypto['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'timestamp' in crypto
                    else crypto.get('lastUpdated', 'N/A')
                )
            } for crypto in crypto_data
        ] if crypto_data else []

        commodity_symbols = ['GC=F', 'CL=F']
        commodity_data = get_fmp_data(f"quote/{','.join(commodity_symbols)}")
        commodity_result = [
            {
                "symbol": commodity['symbol'],
                "price": commodity['price'],
                "change": commodity['change'],
                "volume": commodity['volume'],
                "updated_at": (
                    datetime.fromtimestamp(commodity['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    if 'timestamp' in commodity
                    else commodity.get('lastUpdated', 'N/A')
                )
            } for commodity in commodity_data
        ] if commodity_data else []

        economic_result = []
        gdp_data = get_alpha_vantage_economic("REAL_GDP")
        if gdp_data and len(gdp_data) > 0:
            economic_result.append({
                "indicator": "Real GDP",
                "value": gdp_data[0]["value"],
                "date": gdp_data[0]["date"]
            })
        cpi_data = get_alpha_vantage_economic("CPI")
        if cpi_data and len(cpi_data) > 0:
            economic_result.append({
                "indicator": "CPI",
                "value": cpi_data[0]["value"],
                "date": cpi_data[0]["date"]
            })
        if not economic_result:
            economic_result = [
                {"indicator": "GDP", "value": "2.5%", "date": "2025-03-01"},
                {"indicator": "CPI", "value": "3.1%", "date": "2025-03-01"}
            ]

        finance_data = {
            "stocks": stock_result,
            "etfs": etf_result,
            "cryptos": crypto_result,
            "commodities": commodity_result,
            "economic_indicators": economic_result
        }
        return jsonify(finance_data)
    except Exception as e:
        logger.error(f"Error in finance_data: {str(e)}")
        return jsonify({"error": f"Không thể lấy dữ liệu tài chính: {str(e)}"}), 500

@bp.route('/api/suggest_matrix', methods=['POST'])
def suggest_matrix():
    """Suggest a pairwise comparison matrix."""
    try:
        data = request.get_json()
        loai = data.get('loai')
        name = data.get('name')
        items = data.get('items')
        symbol = data.get('symbol')

        if not items or len(items) < 2:
            return jsonify({"error": "Cần ít nhất 2 mục để so sánh!"}), 400

        n = len(items)
        matrix = np.ones((n, n))

        financial_data = None
        if symbol:
            financial_data = get_fmp_data(f"quote/{symbol}")
            if not financial_data:
                logger.error(f"Cannot fetch financial data for symbol: {symbol}")
                return jsonify({"error": f"Không thể lấy dữ liệu tài chính cho mã {symbol}!"}), 500
            financial_data = financial_data[0]

        for i in range(n):
            for j in range(i + 1, n):
                item1, item2 = items[i].lower(), items[j].lower()
                value = 1
                if financial_data:
                    if "lợi nhuận" in item1 and "rủi ro" in item2:
                        change = financial_data.get('change', 0)
                        if change > 2:
                            value = 3
                        elif change < -2:
                            value = 1/3
                    elif "rủi ro" in item1 and "lợi nhuận" in item2:
                        change = financial_data.get('change', 0)
                        if change > 2:
                            value = 1/3
                        elif change < -2:
                            value = 3
                    elif "thanh khoản" in item1 and "lợi nhuận" in item2:
                        volume = financial_data.get('volume', 0)
                        change = financial_data.get('change', 0)
                        if volume > 1000000 and change < 1:
                            value = 2
                        elif change > 2:
                            value = 1/2
                    elif "lợi nhuận" in item1 and "thanh khoản" in item2:
                        volume = financial_data.get('volume', 0)
                        change = financial_data.get('change', 0)
                        if volume > 1000000 and change < 1:
                            value = 1/2
                        elif change > 2:
                            value = 2
                matrix[i][j] = value
                matrix[j][i] = 1 / value

        return jsonify({"matrix": matrix.tolist()})
    except Exception as e:
        logger.error(f"Error suggesting matrix: {str(e)}")
        return jsonify({"error": f"Lỗi khi gợi ý ma trận: {str(e)}"}), 500

@bp.route('/api/check_consistency', methods=['POST'])
def check_consistency():
    """Check matrix consistency."""
    try:
        data = request.get_json()
        matrix = np.array(data.get('matrix'))
        
        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Ma trận phải là ma trận vuông!"}), 400
        
        ahp_results = perform_ahp_steps(matrix)
        return jsonify({
            "cr": ahp_results['cr'],
            "is_consistent": ahp_results['cr'] < 0.1,
            "message": "Ma trận nhất quán." if ahp_results['cr'] < 0.1 else "Ma trận không nhất quán (CR >= 0.1). Vui lòng điều chỉnh."
        })
    except Exception as e:
        logger.error(f"Error checking consistency: {str(e)}")
        return jsonify({"error": f"Lỗi khi kiểm tra tính nhất quán: {str(e)}"}), 500

@bp.route('/matrix/<loai>/<name>', methods=['GET', 'POST'])
def matrix(loai, name):
    try:
        if loai == 'criteria':
            items = lay_danh_sach_tieu_chi()
        else:
            items = lay_danh_sach_phuong_an()
        
        n = len(items)
        if n < 2:
            flash("Cần ít nhất 2 mục để so sánh!", "danger")
            return redirect(url_for('routes.index'))
        
        item_names = [item['ten'] for item in items]
        existing_matrix = lay_ma_tran_so_sanh(loai, name)
        if existing_matrix and len(existing_matrix['ma_tran']) == n:
            matrix = np.array(existing_matrix['ma_tran'])
        else:
            matrix = np.ones((n, n))
        
        if request.method == 'POST':
            try:
                for i in range(n):
                    for j in range(i + 1, n):
                        key = f"{i}_{j}"  # Sửa lỗi cú pháp từ "{i}_{{j}}" thành "{i}_{j}"
                        value = request.form.get(key, '1')
                        try:
                            matrix[i][j] = float(value)
                            matrix[j][i] = 1 / float(value)
                        except ValueError:
                            flash(f"Giá trị không hợp lệ tại {item_names[i]} vs {item_names[j]}! Vui lòng nhập số hợp lệ.", "danger")
                            return render_template('matrix.html', loai=loai, name=name, items=item_names, matrix=matrix)
                
                ahp_results = perform_ahp_steps(matrix)
                cr = ahp_results['cr']
                
                if cr >= 0.1:
                    flash(f"Ma trận không nhất quán (CR = {cr:.4f} >= 0.1). Vui lòng điều chỉnh!", "warning")
                    return render_template('matrix.html', loai=loai, name=name, items=item_names, matrix=matrix)
                
                # Kiểm tra và chuyển đổi dữ liệu sang danh sách
                ket_qua_ahp = {
                    "ma_tran": matrix.tolist() if isinstance(matrix, np.ndarray) else matrix,
                    "trong_so": ahp_results['trong_so'].tolist() if isinstance(ahp_results['trong_so'], np.ndarray) else ahp_results['trong_so'],
                    "tong_cot": ahp_results['tong_cot'].tolist() if isinstance(ahp_results['tong_cot'], np.ndarray) else ahp_results['tong_cot'],
                    "ma_tran_chuan_hoa": ahp_results['ma_tran_chuan_hoa'].tolist() if isinstance(ahp_results['ma_tran_chuan_hoa'], np.ndarray) else ahp_results['ma_tran_chuan_hoa'],
                    "tong_trong_so": normalize_tong_trong_so(ahp_results['tong_trong_so']),
                    "vector_nhat_quan": ahp_results['vector_nhat_quan'].tolist() if isinstance(ahp_results['vector_nhat_quan'], np.ndarray) else ahp_results['vector_nhat_quan'],
                    "lambda_max": float(ahp_results['lambda_max']),
                    "ci": float(ahp_results['ci']),
                    "cr": float(ahp_results['cr']),
                    "thoi_gian": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                logger.info(f"Attempting to save matrix for loai={loai}, name={name}: {ket_qua_ahp}")
                result = luu_ma_tran_so_sanh(loai, name, ket_qua_ahp)
                if not result:
                    logger.error(f"Failed to save matrix for loai={loai}, name={name}")
                    flash("Không thể lưu ma trận vào cơ sở dữ liệu!", "danger")
                else:
                    logger.info(f"Successfully saved matrix for loai={loai}, name={name}")
                    flash("Lưu ma trận thành công!", "success")
                return redirect(url_for('routes.index'))
            except ValueError as ve:
                logger.error(f"ValueError processing matrix: {str(ve)}")
                flash(f"Lỗi trong tính toán ma trận: {str(ve)}", "danger")
                return render_template('matrix.html', loai=loai, name=name, items=item_names, matrix=matrix)
            except Exception as e:
                logger.error(f"Error processing matrix: {str(e)}")
                flash(f"Lỗi khi lưu ma trận: {str(e)}", "danger")
                return render_template('matrix.html', loai=loai, name=name, items=item_names, matrix=matrix)
        
        return render_template('matrix.html', loai=loai, name=name, items=item_names, matrix=matrix)
    except Exception as e:
        logger.error(f"Error loading matrix page: {str(e)}")
        flash(f"Lỗi khi tải trang nhập ma trận: {str(e)}", "danger")
        return redirect(url_for('routes.index'))
        
@bp.route('/calculate', methods=['POST'])
def calculate():
    """Calculate AHP results."""
    try:
        criteria = lay_danh_sach_tieu_chi()
        alternatives = lay_danh_sach_phuong_an()
        
        if len(criteria) < 1 or len(alternatives) < 2:
            flash("Vui lòng thêm ít nhất 1 tiêu chí và 2 phương án!", "danger")
            return redirect(url_for('routes.index'))
        
        criteria_matrix_doc = lay_ma_tran_so_sanh("criteria", None)
        if not criteria_matrix_doc:
            flash("Vui lòng nhập ma trận so sánh tiêu chí!", "danger")
            return redirect(url_for('routes.index'))
        
        valid, error = validate_matrix_doc(criteria_matrix_doc, "criteria")
        if not valid:
            flash(error, "danger")
            return redirect(url_for('routes.index'))
        
        if len(criteria_matrix_doc['trong_so']) != len(criteria):
            flash("Kích thước ma trận tiêu chí không khớp với số tiêu chí. Vui lòng cập nhật ma trận!", "danger")
            return redirect(url_for('routes.index'))
        
        alt_matrices = []
        for crit in criteria:
            alt_matrix_doc = lay_ma_tran_so_sanh("phuong_an", crit['ten'])
            if not alt_matrix_doc:
                flash(f"Vui lòng nhập ma trận phương án cho {crit['ten']}!", "danger")
                return redirect(url_for('routes.index'))
            valid, error = validate_matrix_doc(alt_matrix_doc, f"phương án {crit['ten']}")
            if not valid:
                flash(error, "danger")
                return redirect(url_for('routes.index'))
            if len(alt_matrix_doc['ma_tran']) != len(alternatives):
                flash(f"Ma trận phương án cho {crit['ten']} không khớp với số phương án. Vui lòng cập nhật!", "danger")
                return redirect(url_for('routes.index'))
            alt_matrices.append(alt_matrix_doc)
        
        criteria_weights = np.array(criteria_matrix_doc['trong_so'])
        final_scores = np.zeros(len(alternatives))
        alt_weights_per_criterion = []
        
        for i, crit in enumerate(criteria):
            alt_matrix = np.array(alt_matrices[i]['ma_tran'])
            alt_weights = calculate_weights(normalize_matrix(alt_matrix)[0])
            alt_weights_per_criterion.append(alt_weights)
            final_scores += criteria_weights[i] * alt_weights
        
        ranking = [{"name": alt['ten'], "score": float(score)} for alt, score in zip(alternatives, final_scores)]
        ranking.sort(key=lambda x: x['score'], reverse=True)
        
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.family'] = 'DejaVuSans' if os.path.exists(FONT_PATH) else 'sans-serif'
        plt.bar([item['name'] for item in ranking], [item['score'] for item in ranking], color='skyblue')
        plt.xlabel('Phương án')
        plt.ylabel('Điểm số')
        plt.title('Xếp hạng đầu tư')
        plt.xticks(rotation=45)
        chart_filename = f"ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        chart_path = os.path.join(CHART_DIR, chart_filename)
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        result_doc = {
            "thoi_gian": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "criteria_matrix": criteria_matrix_doc,
            "criteria_names": [crit['ten'] for crit in criteria],
            "alternative_names": [alt['ten'] for alt in alternatives],
            "alt_matrices": alt_matrices,
            "alt_weights_per_criterion": [weights.tolist() for weights in alt_weights_per_criterion],
            "ranking": ranking,
            "chart": os.path.join('charts', chart_filename).replace('\\', '/')
        }
        result = luu_ket_qua(result_doc)
        if not result:
            logger.error("Failed to save AHP result")
            flash("Không thể lưu kết quả AHP!", "danger")
            return redirect(url_for('routes.index'))
        
        return render_template('result.html', result=result_doc, enumerate=enumerate)
    except Exception as e:
        logger.error(f"Error calculating AHP results: {str(e)}")
        flash(f"Lỗi khi tính toán AHP: {str(e)}", "danger")
        return redirect(url_for('routes.index'))

@bp.route('/weights_explain/<result_id>', methods=['GET'])
def weights_explain(result_id):
    """Explain weights for a result."""
    try:
        result = lay_ket_qua_theo_id(result_id)
        if not result:
            flash("Kết quả không tìm thấy!", "danger")
            return redirect(url_for('routes.index'))
        
        valid, error = validate_matrix_doc(result.get('criteria_matrix', {}), "criteria")
        if not valid:
            flash(error, "danger")
            return redirect(url_for('routes.index'))
        
        criteria_names = result['criteria_names']
        alternative_names = result['alternative_names']
        criteria_weights = result['criteria_matrix']['trong_so']
        alt_weights_per_criterion = result['alt_weights_per_criterion']
        
        explanations = []
        for i, alt_name in enumerate(alternative_names):
            breakdown = f"Phân tích điểm số của '{alt_name}':<br>"
            total_score = 0
            contributions = []
            for j, crit_name in enumerate(criteria_names):
                crit_weight = criteria_weights[j]
                alt_weight = alt_weights_per_criterion[j][i]
                contribution = crit_weight * alt_weight
                total_score += contribution
                contributions.append({
                    "criterion": crit_name,
                    "criteria_weight": crit_weight,
                    "alternative_weight": alt_weight,
                    "contribution": contribution
                })
            breakdown += f"Tổng điểm: {total_score:.4f}"
            explanations.append({
                "alternative": alt_name,
                "total_score": total_score,
                "contributions": contributions,
                "breakdown": breakdown
            })
        
        return render_template('weights_explain.html', explanations=explanations, result_id=result_id)
    except Exception as e:
        logger.error(f"Error explaining weights: {str(e)}")
        flash(f"Lỗi khi giải thích trọng số: {str(e)}", "danger")
        return redirect(url_for('routes.index'))