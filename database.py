
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
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

# Khởi tạo kết nối MongoDB
def get_db():
    """Khởi tạo và trả về kết nối đến cơ sở dữ liệu MongoDB."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['ahp_project_db']
        logger.info("Connected to MongoDB database: ahp_project_db")
        return db
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

# Lấy các collection
def get_collections():
    """Lấy các collection từ cơ sở dữ liệu."""
    try:
        db = get_db()
        collections = {
            "tieu_chi": db['criteria'],
            "phuong_an": db['alternatives'],
            "so_sanh": db['pairwise_comparisons'],
            "ket_qua": db['results']
        }
        logger.info("Retrieved collections: criteria, alternatives, pairwise_comparisons, results")
        return collections
    except Exception as e:
        logger.error(f"Error retrieving collections: {str(e)}")
        raise

# Hàm thêm tiêu chí
def them_tieu_chi(ten):
    """Thêm một tiêu chí mới vào cơ sở dữ liệu."""
    try:
        collections = get_collections()
        result = collections['tieu_chi'].insert_one({"ten": ten})
        logger.info(f"Inserted criterion: {ten}, ID: {result.inserted_id}")
        return result
    except Exception as e:
        logger.error(f"Error inserting criterion {ten}: {str(e)}")
        return None

# Hàm thêm phương án
def them_phuong_an(ten):
    """Thêm một phương án mới vào cơ sở dữ liệu."""
    try:
        collections = get_collections()
        result = collections['phuong_an'].insert_one({"ten": ten})
        logger.info(f"Inserted alternative: {ten}, ID: {result.inserted_id}")
        return result
    except Exception as e:
        logger.error(f"Error inserting alternative {ten}: {str(e)}")
        return None

# Hàm xóa tiêu chí
def xoa_tieu_chi(id_tieu_chi, ten_tieu_chi):
    """Xóa một tiêu chí và các ma trận liên quan."""
    try:
        collections = get_collections()
        collections['tieu_chi'].delete_one({"_id": ObjectId(id_tieu_chi)})
        collections['so_sanh'].delete_many({"ten_tieu_chi": ten_tieu_chi})
        logger.info(f"Deleted criterion ID: {id_tieu_chi}, Name: {ten_tieu_chi}")
    except Exception as e:
        logger.error(f"Error deleting criterion ID: {id_tieu_chi}, Name: {ten_tieu_chi}: {str(e)}")
        raise

# Hàm xóa phương án
def xoa_phuong_an(id_phuong_an):
    """Xóa một phương án."""
    try:
        collections = get_collections()
        collections['phuong_an'].delete_one({"_id": ObjectId(id_phuong_an)})
        logger.info(f"Deleted alternative ID: {id_phuong_an}")
    except Exception as e:
        logger.error(f"Error deleting alternative ID: {id_phuong_an}: {str(e)}")
        raise

# Hàm xóa kết quả
def xoa_ket_qua(id_ket_qua):
    """Xóa một kết quả."""
    try:
        collections = get_collections()
        collections['ket_qua'].delete_one({"_id": ObjectId(id_ket_qua)})
        logger.info(f"Deleted result ID: {id_ket_qua}")
    except Exception as e:
        logger.error(f"Error deleting result ID: {id_ket_qua}: {str(e)}")
        raise

# Hàm lấy danh sách tiêu chí
def lay_danh_sach_tieu_chi():
    """Lấy danh sách tất cả tiêu chí."""
    try:
        collections = get_collections()
        criteria = list(collections['tieu_chi'].find())
        logger.info(f"Retrieved {len(criteria)} criteria")
        return criteria
    except Exception as e:
        logger.error(f"Error retrieving criteria: {str(e)}")
        return []

# Hàm lấy danh sách phương án
def lay_danh_sach_phuong_an():
    """Lấy danh sách tất cả phương án."""
    try:
        collections = get_collections()
        alternatives = list(collections['phuong_an'].find())
        logger.info(f"Retrieved {len(alternatives)} alternatives")
        return alternatives
    except Exception as e:
        logger.error(f"Error retrieving alternatives: {str(e)}")
        return []

# Hàm lấy danh sách kết quả
def lay_danh_sach_ket_qua():
    """Lấy danh sách tất cả kết quả, sắp xếp theo thời gian mới nhất."""
    try:
        collections = get_collections()
        results = list(collections['ket_qua'].find().sort("thoi_gian", -1))
        logger.info(f"Retrieved {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error retrieving results: {str(e)}")
        return []

# Hàm lưu ma trận so sánh
def luu_ma_tran_so_sanh(loai, ten_tieu_chi, ket_qua_ahp):
    """Lưu ma trận so sánh vào cơ sở dữ liệu."""
    try:
        collections = get_collections()
        collection = collections['so_sanh']
        
        # Kiểm tra và chuyển đổi dữ liệu sang danh sách nếu cần
        ma_tran = ket_qua_ahp['ma_tran'] if isinstance(ket_qua_ahp['ma_tran'], list) else ket_qua_ahp['ma_tran'].tolist()
        trong_so = ket_qua_ahp['trong_so'] if isinstance(ket_qua_ahp['trong_so'], list) else ket_qua_ahp['trong_so'].tolist()
        tong_cot = ket_qua_ahp['tong_cot'] if isinstance(ket_qua_ahp['tong_cot'], list) else ket_qua_ahp['tong_cot'].tolist()
        ma_tran_chuan_hoa = ket_qua_ahp['ma_tran_chuan_hoa'] if isinstance(ket_qua_ahp['ma_tran_chuan_hoa'], list) else ket_qua_ahp['ma_tran_chuan_hoa'].tolist()
        vector_nhat_quan = ket_qua_ahp['vector_nhat_quan'] if isinstance(ket_qua_ahp['vector_nhat_quan'], list) else ket_qua_ahp['vector_nhat_quan'].tolist()
        tong_trong_so = float(ket_qua_ahp['tong_trong_so'])  # Đảm bảo là float, không gọi .tolist()
        
        # Ghi log để kiểm tra dữ liệu
        logger.info(f"Saving matrix type={loai}, ten_tieu_chi={ten_tieu_chi}")
        logger.info(f"ma_tran type: {type(ket_qua_ahp['ma_tran'])}, value: {ma_tran[:2]}...")
        logger.info(f"trong_so type: {type(ket_qua_ahp['trong_so'])}, value: {trong_so}")
        logger.info(f"tong_cot type: {type(ket_qua_ahp['tong_cot'])}, value: {tong_cot}")
        logger.info(f"ma_tran_chuan_hoa type: {type(ket_qua_ahp['ma_tran_chuan_hoa'])}, value: {ma_tran_chuan_hoa[:2]}...")
        logger.info(f"vector_nhat_quan type: {type(ket_qua_ahp['vector_nhat_quan'])}, value: {vector_nhat_quan}")
        logger.info(f"tong_trong_so type: {type(ket_qua_ahp['tong_trong_so'])}, value: {tong_trong_so}")
        
        # Cập nhật hoặc chèn mới
        result = collection.update_one(
            {"loai": loai, "ten_tieu_chi": ten_tieu_chi if loai == 'phuong_an' else None},
            {
                "$set": {
                    "ma_tran": ma_tran,
                    "trong_so": trong_so,
                    "tong_cot": tong_cot,
                    "ma_tran_chuan_hoa": ma_tran_chuan_hoa,
                    "tong_trong_so": tong_trong_so,
                    "vector_nhat_quan": vector_nhat_quan,
                    "lambda_max": float(ket_qua_ahp['lambda_max']),
                    "ci": float(ket_qua_ahp['ci']),
                    "cr": float(ket_qua_ahp['cr']),
                    "thoi_gian": ket_qua_ahp['thoi_gian']
                }
            },
            upsert=True
        )
        
        logger.info(f"Saved matrix type={loai}, ten_tieu_chi={ten_tieu_chi}, result={result.modified_count or result.upserted_id}")
        return result.modified_count > 0 or result.upserted_id is not None
    except Exception as e:
        logger.error(f"Error saving matrix type={loai}, ten_tieu_chi={ten_tieu_chi}: {str(e)}")
        return False

# Hàm lấy ma trận so sánh
def lay_ma_tran_so_sanh(loai, ten_tieu_chi):
    """Lấy ma trận so sánh từ cơ sở dữ liệu."""
    try:
        collections = get_collections()
        matrix = collections['so_sanh'].find_one({"loai": loai, "ten_tieu_chi": ten_tieu_chi if loai == 'phuong_an' else None})
        logger.info(f"Retrieved matrix type={loai}, ten_tieu_chi={ten_tieu_chi}")
        return matrix
    except Exception as e:
        logger.error(f"Error retrieving matrix type={loai}, ten_tieu_chi={ten_tieu_chi}: {str(e)}")
        return None

# Hàm lưu kết quả
def luu_ket_qua(ket_qua):
    """Lưu kết quả AHP vào cơ sở dữ liệu."""
    try:
        collections = get_collections()
        result = collections['ket_qua'].insert_one(ket_qua)
        logger.info(f"Inserted result ID: {result.inserted_id}")
        return result
    except Exception as e:
        logger.error(f"Error inserting result: {str(e)}")
        return None

# Hàm lấy kết quả theo ID
def lay_ket_qua_theo_id(id_ket_qua):
    """Lấy kết quả AHP theo ID."""
    try:
        collections = get_collections()
        result = collections['ket_qua'].find_one({"_id": ObjectId(id_ket_qua)})
        logger.info(f"Retrieved result ID: {id_ket_qua}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving result ID: {id_ket_qua}: {str(e)}")
        return None
