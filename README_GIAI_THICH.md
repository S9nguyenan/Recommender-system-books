# Đọc hiểu dự án khuyến nghị sách — dành cho người mới

Tài liệu này giải thích mã nguồn đang có trong dự án, từ lúc nhập lệnh đến lúc nhận kết quả. Anh có thể mở song song file này và file Python được nhắc tới để đọc từng phần. README gốc vẫn giữ vai trò hướng dẫn cài đặt và chạy nhanh.

Các ví dụ số nhỏ trong tài liệu chỉ để giải thích phép tính, không phải kết quả thực nghiệm. Kết quả thực tế nằm trong `results/`.

## 1. Trước tiên, dự án đang giải quyết bài toán gì?

Đầu vào là danh sách sách và lịch sử người dùng tương tác, chấm điểm sách. Đầu ra là một danh sách sách được sắp theo mức phù hợp với người dùng.

Ví dụ: một người từng chấm cao vài cuốn sách. Chương trình dùng lịch sử đó để chọn ra 10 cuốn khác nên đề xuất cho người ấy. Mỗi thuật toán có một cách tính độ phù hợp riêng.

Dự án có hai nhiệm vụ liên quan nhưng khác nhau:

| Nhiệm vụ | Câu hỏi cần trả lời | Ví dụ đầu ra |
|---|---|---|
| Dự đoán điểm | Người A có thể chấm sách B bao nhiêu điểm? | 8,2 trên thang 1–10 |
| Khuyến nghị Top-N | Nên đưa những sách nào lên đầu cho người A? | Danh sách 10 sách |

User-CF, Item-CF và SVD làm cả hai. Popularity, Content-Based và LightFM trong dự án chỉ cung cấp điểm xếp hạng, không được đánh giá bằng MAE/RMSE.

Đây là chương trình thực nghiệm chạy trong terminal. Hiện chưa có giao diện, đăng nhập, API hay cơ sở dữ liệu; dữ liệu được đọc trực tiếp từ CSV. Sau khi chạy xong, chương trình lưu các bảng kết quả, chưa lưu mô hình để nạp lại lần sau.

### Những từ sẽ gặp nhiều

| Từ trong code | Nghĩa dễ hiểu |
|---|---|
| user | Người dùng |
| item / book | Sách |
| interaction | Một cặp người dùng–sách có ghi nhận tương tác |
| rating | Điểm đánh giá |
| catalog | Danh mục sách được phép xét |
| metadata | Thông tin mô tả sách: tên, tác giả, nhà xuất bản… |
| model | Mô hình: đối tượng chứa dữ liệu và cách tính để đề xuất sách |
| train | Phần dữ liệu cho mô hình học |
| test | Phần dữ liệu giữ lại để chấm chất lượng |
| fit | Chuẩn bị/học mô hình từ train; không phải mô hình nào cũng có vòng lặp tối ưu |
| predict | Dự đoán điểm cho một cặp user–book |
| score_items | Tính điểm xếp hạng cho các sách trong catalog |
| recommend | Lấy các sách điểm cao nhất sau khi lọc sách đã thấy |
| baseline / fallback | Cách tính dự phòng khi thiếu dữ liệu |
| cold-start | Người dùng hoặc sách chưa có đủ lịch sử để mô hình tính như bình thường |
| backend | Phần triển khai bên dưới, ví dụ SVD dùng Surprise hoặc NumPy |
| seed | Số khởi tạo ngẫu nhiên để dễ lặp lại thí nghiệm |

## 2. Toàn bộ cấu trúc thư mục

Cây dưới đây liệt kê các file của dự án theo nhóm. Các thư mục do Python/thư viện tự sinh được gom ở cuối; không liệt kê hàng nghìn file nội bộ của thư viện.

```text
Khuyến nghị sách/
├── README.md
├── README_GIAI_THICH.md          ← tài liệu anh đang đọc
├── .gitignore
├── requirements.txt
├── requirements-optional.txt
├── environment.yml
├── setup_environment.py
├── run.py
├── data/
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── popularity.py
│   │   ├── content_based.py
│   │   ├── cf_common.py
│   │   ├── user_based_cf.py
│   │   ├── item_based_cf.py
│   │   ├── svd.py
│   │   └── lightfm_model.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── evaluate.py
├── experiments/
│   ├── __init__.py
│   ├── check_data.py
│   ├── runner.py
│   ├── run_all.py
│   ├── run_popularity.py
│   ├── run_content_based.py
│   ├── run_user_cf.py
│   ├── run_item_cf.py
│   ├── run_svd.py
│   └── run_lightfm.py
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_real_models.py
├── results/
│   ├── data_validation.csv
│   ├── data_validation.json
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   └── quick_20261018T085504340706Z/
│       ├── manifest.json
│       ├── model_comparison.csv
│       ├── model_comparison.json
│       └── các file kết quả riêng của 6 mô hình (giải thích ở mục 10)
├── .venv/                       ← môi trường Python cũ trong dự án
├── .cache/                      ← bộ nhớ đệm, công cụ và gói đã tải
└── __pycache__/                 ← có thể xuất hiện trong nhiều thư mục
```

Tên thư mục kết quả sẽ tăng lên sau các lần chạy mới. Những thư mục lịch sử cũ đã được dọn; cây này mô tả phần còn lại lúc viết tài liệu.

### Các file `__init__.py` là gì?

Các file này có trong `src/`, `src/utils/`, `src/data_processing/`, `src/models/`, `src/evaluation/`, `experiments/` và `tests/`. Chúng đánh dấu các thư mục là gói Python thông thường, giúp tổ chức đường dẫn import như `from src.models.svd import SVDRecommender`. Chúng không chứa thuật toán khuyến nghị; không cần bắt đầu đọc từ đây.

## 3. Các file ở thư mục gốc: khởi động và môi trường

### `README.md` và `README_GIAI_THICH.md`

`README.md` chứa hướng dẫn chạy, mô tả và lịch sử thực nghiệm. `README_GIAI_THICH.md` giải thích cách code hoạt động. Cả hai là tài liệu, chương trình không đọc chúng khi huấn luyện.

### `.gitignore`

Khai báo những file/thư mục Git nên bỏ qua, như `.venv/`, `.cache/`, `__pycache__/`, `*.pyc`. Git dùng file này để hạn chế đưa môi trường và cache vào quản lý mã nguồn. Nó không điều khiển thuật toán.

### `requirements.txt`

Danh sách thư viện nền để cài bằng pip:

| Thư viện | Vai trò trong dự án |
|---|---|
| NumPy | Mảng số, vector, phép nhân, tính sai số, backend SVD tự viết |
| Pandas | Đọc/ghi CSV và xử lý bảng dữ liệu bằng DataFrame |
| SciPy | Ma trận thưa: lưu ít bộ nhớ khi đa số user chưa đánh giá đa số sách |
| scikit-learn | TF-IDF và chuẩn hóa vector |

`DataFrame` có thể hiểu là bảng gồm hàng và cột như Excel, nhưng được xử lý bằng Python.

### `requirements-optional.txt`

Liệt kê `scikit-surprise` và `lightfm`. Surprise cung cấp SVD; LightFM cung cấp mô hình LightFM. Từ “optional” nghĩa là phần thư viện biên dịch được tách riêng để lỗi cài chúng không chặn các mô hình khác. Muốn chạy LightFM thật thì vẫn cần cài LightFM. SVD có backend NumPy dự phòng khi không import được Surprise.

### `environment.yml`

Cấu hình môi trường Conda: Python 3.11, các thư viện và khoảng phiên bản tương thích, nguồn gói `conda-forge`. Đây là bộ cấu hình dùng để cài đủ mô hình trên máy này; nó không phải danh sách dữ liệu đầu vào.

### `setup_environment.py`

File cài môi trường, không phải file huấn luyện. Trên Windows, `main()` làm các việc:

1. Kiểm tra đã có `%LOCALAPPDATA%/BookRecommender/env311/python.exe` chưa; có thì dừng cài.
2. Chép cấu hình môi trường vào đường dẫn không dấu dưới `%LOCALAPPDATA%/BookRecommender`.
3. Nếu thiếu micromamba, tải công cụ này vào `.cache/micromamba/`.
4. Gọi micromamba để tạo môi trường từ `environment.yml`.

Micromamba là công cụ cài môi trường và các gói Conda. Môi trường đầy đủ nằm ngoài thư mục dự án; `.venv/` là môi trường cũ, không phải nơi `run.py` đang ưu tiên chạy.

### `run.py` — cổng vào khi chạy lệnh

Khi anh gõ:

```powershell
python run.py -m experiments.run_all --mode quick
```

Python hiện tại chạy `run.py`. Hàm `main()` tìm Python 3.11 của môi trường đầy đủ, bổ sung các thư mục DLL Conda vào PATH của tiến trình con, rồi dùng `subprocess.call()` gọi Python đó với phần lệnh phía sau.

`-m experiments.run_all` nghĩa là “chạy module `experiments/run_all.py`”. Anh không cần mở file đó rồi bấm Run từng dòng. Nếu không truyền tham số, `python run.py` mặc định chạy cả 6 mô hình ở chế độ quick. Nếu thiếu môi trường đầy đủ, `run.py` báo lỗi và yêu cầu cài, không tự chuyển sang `.venv`.

## 4. `data/`: nguyên liệu đầu vào

### `Books.csv`

Mỗi dòng mô tả một sách. Các cột gồm `Book-ID`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, `Image-URL-S`, `Image-URL-M`, `Image-URL-L`.

`Book-ID` là khóa để nối sách với bảng Ratings. Tên, tác giả và nhà xuất bản được Content-Based dùng làm đặc trưng. Các URL ảnh chỉ là dữ liệu mô tả, chương trình không tải ảnh để học. Năm xuất bản được làm sạch nhưng hiện không được đưa vào đặc trưng của 6 mô hình.

### `Ratings.csv`

Mỗi dòng có `User-ID`, `Book-ID`, `Book-Rating`, cho biết một user đã có tương tác với một sách.

| Giá trị `Book-Rating` | Ý nghĩa trong dự án |
|---|---|
| 1–10 | Điểm tường minh: người dùng thực sự chấm điểm |
| 0 | Tương tác ngầm: có ghi nhận tương tác, không phải chấm sách 0 điểm |
| Không có dòng user–book | Chưa quan sát tương tác; khác với một dòng có rating 0 |

Điểm 0 không tham gia MAE/RMSE và không được coi là điểm thấp trong CF/SVD. Riêng LightFM coi tương tác 0 là nhãn dương theo chính sách của dự án.

### `Users.csv`

Chứa `User-ID`, `Location`, `Age`. Tiền xử lý dùng bảng này để xác nhận user hợp lệ và làm sạch tuổi/vị trí. Hiện các mô hình không dùng tuổi hoặc vị trí để tính độ phù hợp.

## 5. `src/utils/config.py`: các tham số mặc định

File này không huấn luyện mô hình. Nó tập trung đường dẫn và tham số để các nơi khác import dùng chung.

`PROJECT_ROOT` được tính từ vị trí file; `DATA_DIR` trỏ đến `data/`, `RESULTS_DIR` trỏ đến `results/`. Vì thế đường dẫn mặc định không phụ thuộc anh đang đứng ở thư mục terminal nào khi gọi module trực tiếp.

`PreprocessingConfig` quản lý quy tắc làm sạch: năm 1450–2026, tuổi 5–100, tỷ lệ test 0,2 và seed 42. Tuổi/năm ngoài khoảng được đặt thành thiếu; không phải cứ thiếu tuổi là loại người dùng.

`ExperimentConfig` quản lý thực nghiệm:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `mode` | `quick` | Chọn phạm vi dữ liệu |
| `seed` | 42 | Điều khiển tính ngẫu nhiên |
| `max_users` | 1.500 | Giới hạn chọn user trong quick |
| `max_items` | 5.000 | Giới hạn chọn sách trong quick |
| `eval_users` | 50 | Tối đa số user dùng đánh giá Top-K |
| `eval_ratings` | 1.000 | Tối đa số rating test dùng tính MAE/RMSE |
| `neighbors` | 30 | Số láng giềng tối đa của CF |
| `factors` | 32 | Số chiều vector ẩn của SVD/LightFM |
| `epochs` | 15 | Số lượt học của SVD/LightFM |
| `k` | 10 | Số sách đề xuất khi đánh giá |
| `relevance_threshold` | 7 | Rating từ mức này trở lên được xem là thích/liên quan |
| `max_features` | 30.000 | Số đặc trưng từ vựng tối đa của TF-IDF |
| `svd_backend` | `auto` | Ưu tiên Surprise, dự phòng NumPy khi import không được |
| `lightfm_loss` | `logistic` | Hàm mất mát dùng khi học LightFM |

Trong CLI, dấu gạch dưới đổi thành gạch ngang, ví dụ `--eval-users 10`. Các giới hạn user/item/đánh giá nhận 0 để bỏ giới hạn; `k`, số láng giềng, số nhân tố và số epoch phải dương. `__post_init__()` kiểm tra các điều kiện này khi tạo cấu hình.

## 6. `src/data_processing/`: từ CSV đến train/test

### `load_data.py` — đọc dữ liệu

`SCHEMAS` liệt kê những cột bắt buộc cho mỗi bảng. `load_csv(path, required_columns)` dùng `pandas.read_csv()` để đọc UTF-8, ban đầu giữ dữ liệu dạng chuỗi để không mất số 0 đầu mã sách. Thiếu cột, sai mã hóa hoặc lỗi phân tích CSV thì báo lỗi, không âm thầm bỏ dòng lỗi.

`BookCrossingData` là hộp chứa ba bảng `books`, `ratings`, `users`. `load_data(data_dir)` gọi `load_csv()` cho cả ba file rồi trả hộp này. Nó chưa chia train/test và không ghi đè CSV gốc.

### `preprocessing.py` — làm sạch và chia tập

Các hàm nhỏ hỗ trợ:

| Hàm | Công việc |
|---|---|
| `_normalize()` | Bỏ khoảng trắng đầu/cuối chuỗi, đổi chuỗi rỗng thành thiếu |
| `_integer()` | Chuyển số nguyên hợp lệ sang kiểu số nguyên có thể chứa giá trị thiếu |
| `_unique_keys()` | Bỏ dòng trùng hoàn toàn; báo lỗi nếu cùng khóa nhưng nội dung xung đột |

`preprocess(data)` gọi các bước này để:

1. Làm sạch Books, kiểm tra khóa sách, xử lý năm không hợp lệ.
2. Gắn `Metadata-Needs-Review` cho một số dòng có trường năm không chuyển được thành số nguyên; bỏ metadata tác giả/nhà xuất bản/ảnh không đáng tin của những dòng này, giữ tên sách.
3. Kiểm tra khóa user, làm sạch tuổi và vị trí.
4. Giữ các rating nguyên trong 0–10 và khóa user hợp lệ.
5. Loại tương tác có sách/user không thuộc danh mục hợp lệ.
6. Tách tương tác tường minh và ngầm, ghi thống kê vào báo cáo.

Đầu ra `PreparedData` chứa `books`, `users`, `interactions`, `explicit_ratings`, `implicit_interactions`, `report`. Đây là dữ liệu trong bộ nhớ; các file CSV gốc không bị sửa.

`split_interactions()` chia toàn bộ tương tác theo từng user, sau khi xáo trộn với seed. Với user có n tương tác, số dòng test là `min(max(floor(n × 0,2), 1), n − 1)`. Vì vậy user chỉ có một tương tác nằm hoàn toàn trong train; user từ hai tương tác có ít nhất một dòng train và một dòng test.

Ví dụ user có 10 tương tác: thường 8 dòng train, 2 dòng test. Chia chung cả rating 0 và rating 1–10 trước, rồi mô hình mới lấy phần phù hợp để học. Cùng cặp user–book không được xuất hiện ở cả hai tập.

**Vì sao phải giấu test?** Nếu đã cho mô hình học việc user thích sách B rồi lại hỏi nó có đoán được B không, phép đánh giá sẽ dễ một cách sai lệch. Test đóng vai trò phần đáp án giữ kín. Dữ liệu không có timestamp, nên phép chia này là ngẫu nhiên theo user, không mô phỏng thứ tự quá khứ–tương lai.

## 7. `src/models/`: phần tính toán khuyến nghị

### 7.1. `base.py` — lớp nền dùng chung

`BaseRecommender` gom các bước mà nhiều mô hình đều cần. Khi một lớp khác kế thừa lớp này, nó dùng lại các hàm chung và tự bổ sung cách chấm điểm.

`_setup(books, interactions)` nhận **catalog và train**, kiểm tra dữ liệu rồi tạo:

| Thuộc tính | Nội dung |
|---|---|
| `books`, `items` | Bảng sách và danh sách mã sách theo thứ tự chung |
| `item_index` | Ánh xạ mã sách sang vị trí số trong mảng |
| `train` | Bản sao tương tác huấn luyện |
| `seen` | Với mỗi user, tập sách đã tương tác trong train, kể cả rating 0/thấp |
| `explicit` | Chỉ các dòng train có rating 1–10 |
| `global_mean` | Trung bình rating tường minh toàn train; không có thì 5,5 |
| `item_means`, `user_means` | Trung bình điểm theo sách và theo user |
| `rating_counts` | Số rating tường minh train của từng sách |
| `popularity` | Điểm Bayesian của từng sách, giải thích ở phần tiếp theo |
| `trained_items` | Tập sách có bất kỳ tương tác train nào |

`baseline(user_id, book_id)` ưu tiên trung bình của sách; nếu thiếu thì trung bình của user; nếu vẫn thiếu thì trung bình toàn train. Đây là cách dự phòng của CF, không phải công thức cold-start riêng của SVD.

`recommend(user_id, n=10, exclude_seen=True)` gọi `self.score_items(user_id)`. Từ `self` chỉ chính mô hình đang chạy: cùng một hàm `recommend()`, nhưng `score_items()` có thể là cách tính của Content-Based, User-CF hoặc mô hình khác.

Sau đó `_result()` loại sách cần loại bằng điểm âm vô cực, chỉ giữ điểm hữu hạn, sắp giảm dần và lấy n cuốn đầu. Hòa điểm thì theo thứ tự catalog. Đầu ra có `Book-ID`, `Book-Title`, `Book-Author`, `rating_count`, `score`.

`supports_ratings = False` mặc định cho biết mô hình chỉ xếp hạng. CF và SVD đổi thành `True` để bộ đánh giá biết có thể gọi `predict()`.

### 7.2. `popularity.py` — gợi ý dựa vào điểm chung của sách

Lớp: `PopularityRecommender`. `fit()` chỉ gọi `_setup()`, `score_items()` trả lại mảng `popularity` đã tính. Không có vòng lặp học tham số.

Công thức trong `base.py`:

```text
score = (count × mean + 10 × global_mean) / (count + 10)
```

`count` là số rating tường minh của sách trong train; `mean` là điểm trung bình của sách; `global_mean` là trung bình toàn train. Số 10 là mức kéo về trung bình chung, đang được viết cố định trong code.

Ví dụ trung bình chung là 7. Sách A chỉ có một người chấm 10: điểm điều chỉnh là `(1×10 + 10×7)/11 ≈ 7,27`. Sách B có 100 người chấm trung bình 8: điểm là `(100×8 + 10×7)/110 ≈ 7,91`. Sách B đứng trên vì mức điểm tốt có nhiều bằng chứng hơn.

Tên Popularity ở đây không có nghĩa đơn thuần lấy sách nhiều lượt tương tác nhất. Nó dùng trung bình rating có điều chỉnh. Mọi user có cùng điểm sách ban đầu, nhưng danh sách cuối có thể khác vì mỗi người bị loại những sách đã thấy khác nhau. Sách chưa có rating tường minh nhận trung bình chung.

### 7.3. `content_based.py` — gợi ý theo nội dung mô tả

Lớp: `ContentBasedRecommender`. Ý tưởng: tìm sách có mô tả giống những sách user từng thích.

**Trong `fit()`:**

1. Gọi `_setup()` để lấy train và thống kê chung.
2. Ghép `Book-Title`, `Book-Author`, `Publisher` thành một chuỗi cho mỗi sách.
3. Gọi `TfidfVectorizer.fit_transform()` để biến các chuỗi thành ma trận số `features`.
4. Lấy các sách user chấm từ 7 trở lên trong train, gom thành `liked` theo user.

Metadata catalog được dùng để tạo từ vựng, còn rating test không tham gia tạo hồ sơ thích. Code hiện không dùng thể loại, tóm tắt nội dung, ảnh, năm xuất bản hay mô hình hiểu ngôn ngữ.

**TF-IDF là gì?** Mỗi sách được biểu diễn thành một vector, tức một dãy số. Mỗi chiều ứng với một từ. Từ xuất hiện trong sách được tính trọng số; từ xuất hiện ở rất nhiều sách được giảm sức phân biệt.

Với cấu hình mặc định đang dùng của vectorizer, có thể đọc phép tính là:

```text
TF(từ, sách) = số lần từ xuất hiện trong chuỗi của sách
IDF(từ) = ln((1 + số sách) / (1 + số sách có từ đó)) + 1
trọng số ban đầu = TF × IDF
vector cuối = vector trọng số / độ dài vector
```

Ví dụ hai sách cùng tên tác giả có thể có vector gần nhau vì trùng từ tác giả. Từ phổ biến ở hầu hết sách giúp phân biệt ít hơn. Đây là so khớp từ vựng: hai từ khác nhau nhưng đồng nghĩa không tự động được hiểu là cùng ý.

**Trong `score_items(user_id)`:** lấy trung bình vector của các sách user thích, rồi chuẩn hóa thành vector hồ sơ user. Mọi sách được chấm theo độ giống với hồ sơ đó.

```text
cosine(a, b) = tổng(a_j × b_j) / (độ dài a × độ dài b)
```

Hai vector đã chuẩn hóa có độ dài 1 nên cosine chính là tích vô hướng, được tính bằng `features @ profile.T`. Điểm càng cao thì càng giống theo biểu diễn từ vựng. Điểm 0,8 ở đây là mức tương đồng, không phải rating 0,8/10.

Ví dụ user thích sách A và B: hồ sơ bằng trung bình vector A, B. Sách C gần hồ sơ hơn sách D thì C được xếp trên. Các sách đã có bất kỳ tương tác train nào của user vẫn bị loại ở bước `recommend()`.

Nếu user chưa có sách chấm từ ngưỡng trở lên, mô hình dùng điểm Popularity. Các sách đã thích được lấy trung bình ngang nhau; rating 10 không được nhân trọng số lớn hơn rating 7.

`similar_books(book_id, n)` là chức năng khác: so vector một sách với toàn catalog và loại chính sách đó. Nó không cần hồ sơ user; mã sách không tồn tại trong catalog đang chạy thì báo lỗi.

### 7.4. `cf_common.py` — nền tảng cho hai mô hình lọc cộng tác

CF viết tắt của Collaborative Filtering, nghĩa là lọc cộng tác: dùng mẫu đánh giá của cộng đồng thay vì mô tả sách.

Lớp `SparseCF` kế thừa `BaseRecommender`. `fit()` gọi `_setup()`, lấy rating tường minh train rồi tạo ma trận `matrix`:

```text
                 Sách A  Sách B  Sách C
User 1              8       9       .
User 2              7       .       8
User 3              .       5       6
```

Dấu chấm nghĩa là chưa có rating tường minh. Trong ma trận thưa nó không được lưu như một quan sát, không phải nhãn “user ghét sách này”. Các dòng rating 0 cũng không được đưa vào ma trận rating CF.

`csr_matrix` chỉ lưu các phần tử có dữ liệu cùng chỉ số, giúp tiết kiệm bộ nhớ. `observed` là ma trận cùng vị trí nhưng giá trị bằng 1, để tính tổng trọng số chỉ trên người thật sự chấm sách.

`user_norm` chuẩn hóa vector từng user; `item_norm` chuẩn hóa vector từng sách sau khi chuyển vị. `_top()` lấy tối đa 30 độ tương đồng dương cao nhất, có thể loại chính user đang xét. `predict()` lấy điểm sách từ `score_items()` và giới hạn trong 1–10; sách ngoài catalog dùng baseline.

Hai CF hiện dùng cosine của rating gốc, không trừ trung bình từng user trước khi so sánh. Vì vậy đây không phải phiên bản Pearson hay CF có hiệu chỉnh thiên hướng chấm điểm.

### 7.5. `user_based_cf.py` — tìm người có cách chấm giống nhau

Lớp: `UserBasedCF`, kế thừa `SparseCF`. File này chỉ cần viết cách `score_items()`; phần fit và predict dùng lại từ lớp cha.

Luồng tính cho user U:

1. Nếu có kết quả trong `_cache`, dùng lại.
2. Tạo điểm baseline cho mọi sách để có sẵn phương án dự phòng.
3. Tính cosine giữa vector U và từng user khác bằng `user_norm @ user_norm[uid].T`.
4. Loại U, chọn tối đa 30 user tương đồng dương cao nhất.
5. Tính trung bình rating có trọng số cho từng sách, chỉ dùng láng giềng đã chấm sách đó.

```text
dự đoán(U, sách I)
  = tổng(similarity(U,V) × rating(V,I))
    / tổng(similarity(U,V))
```

V chạy qua những láng giềng được chọn và có rating I. Ví dụ V1 giống U ở mức 0,8, chấm I là 9; V2 giống ở mức 0,4, chấm I là 6. Dự đoán là `(0,8×9 + 0,4×6)/(0,8+0,4) = 8`.

Nếu không có láng giềng chấm một sách, điểm baseline của sách đó được giữ nguyên. User chưa có rating tường minh train cũng dùng baseline. Code tính một hàng tương đồng khi cần, không tạo ma trận toàn bộ user×user. Cache giữ tối đa 8 vector điểm user để tránh tốn bộ nhớ quá mức.

Điểm cần đọc đúng: 30 láng giềng được chọn chung cho user trước; không chọn lại 30 người riêng cho từng sách.

### 7.6. `item_based_cf.py` — tìm sách được cộng đồng chấm giống nhau

Lớp: `ItemBasedCF`, cũng kế thừa `SparseCF`. Nó dùng cùng dữ liệu và cosine nhưng so **sách với sách**.

Luồng tính cho user U:

1. Lấy lịch sử rating tường minh của U từ `matrix`.
2. Với mỗi sách ứng viên I, tính cosine giữa I và các sách trong lịch sử đó.
3. Giữ tối đa 30 sách lịch sử giống I nhất.
4. Lấy rating U đã chấm các sách lịch sử để dự đoán điểm I.

```text
dự đoán(U,I)
  = tổng(similarity(I,J) × rating(U,J))
    / tổng(similarity(I,J))
```

Ví dụ U chấm sách A là 9, B là 6. Sách mới C giống A ở mức 0,8, giống B ở mức 0,2. Điểm C bằng `(0,8×9 + 0,2×6)/(0,8+0,2) = 8,4`.

Độ giống ở đây đến từ cách cộng đồng chấm sách, không phải trùng tên/tác giả như Content-Based. Code tính từng block 256 sách ứng viên so với lịch sử U, không tạo một ma trận sách×sách đầy đủ. Nếu tổng độ giống bằng 0, giữ baseline. Cache cũng giới hạn 8 user.

**Cách phân biệt:** User-CF hỏi “người giống anh chấm sách này thế nào?”, Item-CF hỏi “anh đã chấm những sách giống sách này thế nào?”.

### 7.7. `svd.py` — học sở thích dưới dạng các vector ẩn

Lớp: `SVDRecommender`. Mô hình dùng rating tường minh train. Không dùng rating 0 làm nhãn học, không điền tất cả ô trống thành điểm 0.

Tên SVD trong dự án chỉ mô hình phân rã ma trận có thiên lệch, học trên rating quan sát bằng tối ưu lặp. Nó không gọi `numpy.linalg.svd()` để phân rã trực tiếp một bảng đã điền đủ số.

Mỗi user được gán một vector `p`, mỗi sách một vector `q`, mặc định 32 chiều. Những chiều này được học từ rating, không có nhãn cố định kiểu “trinh thám”, “lãng mạn”. Có thể hình dung chúng là các hướng sở thích mà mô hình tự tìm.

```text
điểm dự đoán = μ + b_user + b_item + p_user · q_item
```

Trong đó μ là trung bình toàn train; `b_user` mô tả xu hướng user chấm cao/thấp; `b_item` mô tả xu hướng sách được chấm cao/thấp; tích vô hướng hai vector mô tả mức phù hợp riêng của cặp user–book.

Ví dụ μ=7, user có thiên lệch +0,5, sách có thiên lệch +0,3, phần tương hợp vector là +0,4 thì dự đoán bằng 8,2.

**Trong `fit()`:** gọi `_setup()`, kiểm tra có rating tường minh rồi chọn backend:

| Backend | Cách code chạy |
|---|---|
| `surprise` | Dùng `Dataset`, `Reader(rating_scale=(1,10))`, `SVD` của thư viện Surprise |
| `numpy` | Tự học các vector và bias bằng NumPy |
| `auto` | Thử import Surprise; chỉ khi gặp `ImportError` mới chuyển NumPy |

Nhánh Surprise gọi `Dataset.load_from_df()` với ba cột user, sách, rating; `build_full_trainset()` ở đây chỉ xây từ **phần train đã truyền vào**, không lén ghép test. Sau đó `self.model.fit()` học mô hình. `SURPRISE_DATA_FOLDER` được đặt vào `.cache/surprise` nếu chưa có cấu hình khác.

Nhánh NumPy khởi tạo vector nhỏ ngẫu nhiên, bias bằng 0. Mỗi epoch xáo trộn rating, xử lý từng nhóm 256 dòng rồi sửa tham số theo sai số:

```text
error = rating_thật − rating_dự_đoán
bias ← bias + learning_rate × (error − regularization × bias)
p ← p + learning_rate × (error × q − regularization × p)
q ← q + learning_rate × (error × p_cũ − regularization × q)
```

Đây là ý tưởng cập nhật; code dùng `np.add.at()` cộng dồn đóng góp của các chỉ số user/item lặp trong cùng nhóm. Learning rate mặc định 0,005 điều khiển bước sửa; regularization 0,02 kéo tham số bớt lớn để hạn chế học quá sát train. Mặc định học 15 epoch.

`predict()` trả rating giới hạn 1–10. Với user/sách chưa được học, mô hình chỉ dùng trung bình và các bias đã biết; không lấy vector ngẫu nhiên chưa học để đoán. `score_items()` tính dự đoán cho toàn catalog rồi `recommend()` lọc/sắp xếp.

SVD học để giảm sai số rating; điều này không bảo đảm đứng đầu về Precision@10. Dự đoán điểm và chọn đúng các sách đầu danh sách là hai mục tiêu khác nhau.

### 7.8. `lightfm_model.py` — học mức phù hợp từ nhãn dương/âm

Lớp: `LightFMRecommender`. Dự án gọi thư viện LightFM thật. Cấu hình mặc định dùng **logistic** và identity features, tức nhận diện user/sách bằng ID. Hiện chưa truyền đặc trưng tác giả, nội dung, tuổi hay vị trí vào LightFM; vì vậy chưa phải bản kết hợp nội dung và hành vi dù thư viện có khả năng đó.

**Trong `fit()`:**

1. Import `LightFM`; nếu thiếu thư viện thì báo lỗi.
2. Trên Windows, từ chối loss khác logistic vì bản cài đã có lịch sử crash với các loss xếp hạng.
3. Gọi `_setup()`, lấy dữ liệu train.
4. Đổi rating thành nhãn theo bảng dưới, tạo ma trận thưa `coo_matrix`.
5. Tạo `LightFM(no_components=32, loss='logistic', random_state=42)` và gọi `fit()` với số epoch cấu hình, một thread.

| Tương tác train | Nhãn logistic |
|---|---:|
| Rating 0 | +1 |
| Rating từ 7 trở lên | +1 |
| Rating 1–6 | −1 |
| Cặp chưa quan sát | Không đưa vào dữ liệu học |

Coi rating 0 là dương là một giả định của dự án: tương tác có thể là tín hiệu quan tâm, không chứng minh chắc chắn user thích. Rating 1 và 6 cùng trở thành nhãn −1; độ lớn rating không còn được giữ như trong SVD.

LightFM học vector và bias để tăng điểm cho cặp nhãn dương, giảm điểm cho cặp nhãn âm. Dạng điểm có thể hiểu là:

```text
s = bias_user + bias_item + vector_user · vector_item
loss(y,s) = ln(1 + exp(−y × s)), với y thuộc {−1,+1}
```

Nếu y=+1 mà s thấp thì loss lớn, mô hình cần tăng mức phù hợp; nếu y=−1 mà s cao thì cần giảm. Chi tiết tối ưu do thư viện thực hiện. Dự án lưu score thô để xếp hạng, không coi score là rating 1–10 hoặc xác suất đã được kiểm chứng.

`score_items()` gọi `model.predict()` cho mọi ID sách. Tuy thư viện đặt tên là predict, đầu ra dùng ở đây là điểm xếp hạng. Code loại các sách không có positive train bằng âm vô cực. Đây là chính sách lọc của bản triển khai: sách chỉ có nhãn âm có thể đã tham gia học logistic nhưng vẫn bị loại khỏi đề xuất.

User không tồn tại trong mô hình hoặc không có positive train được chuyển sang Popularity. Các sách đã thấy tiếp tục bị loại bởi hàm chung. Không có negative giả lấy từ test; cặp chưa quan sát cũng không được tự sinh thành nhãn âm trong cấu hình logistic hiện tại.

## 8. `src/evaluation/`: chương trình chấm mô hình thế nào?

### `metrics.py` — các công thức

`rating_metrics(actual, predicted)` so sánh rating thật với dự đoán. Chỉ lấy rating thật trong 1–10 và hai giá trị đều hữu hạn; loại rating 0 và dự đoán NaN/vô cực.

```text
MAE  = tổng |dự đoán − thực tế| / số dòng hợp lệ
RMSE = căn bậc hai(tổng (dự đoán − thực tế)² / số dòng hợp lệ)
```

Ví dụ điểm thật [8,6], dự đoán [7,8]: sai số tuyệt đối [1,2], MAE=1,5; RMSE=√((1²+2²)/2)≈1,58. Cả hai càng thấp càng tốt; RMSE nhạy hơn với các lỗi lớn. `rating_evaluated` cho biết bao nhiêu dòng được tính. Không có dòng hợp lệ thì trả `None`, không trả 0 gây hiểu nhầm là hoàn hảo.

`precision_recall_at_k(recommended, relevant, k)` đếm sách được đề xuất trúng tập sách liên quan:

```text
Precision@K = số sách trúng trong Top-K / K
Recall@K    = số sách trúng trong Top-K / tổng sách liên quan của user trong test
```

Ví dụ test ghi nhận user thích 5 sách; danh sách Top-10 trúng 2 cuốn thì Precision@10=0,2 và Recall@10=0,4. Nếu chỉ trả 6 sách trong yêu cầu Top-10, Precision vẫn chia 10. Sách đề xuất trùng được bỏ trùng trước khi tính. User không có sách liên quan thì hàm trả `(None, None)`.

### `evaluate.py` — chọn mẫu và gọi mô hình

`evaluation_cohorts()` tạo hai nhóm dùng chung cho mọi mô hình trong một lượt chạy:

- Rating test tường minh để tính MAE/RMSE, lấy mẫu theo seed nếu vượt giới hạn.
- User có ít nhất một sách test rating từ ngưỡng trở lên, lấy mẫu theo seed nếu vượt giới hạn.

Giá trị mặc định riêng trong định nghĩa hàm là 100 user/2.000 rating, nhưng runner truyền cấu hình thực nghiệm 50 user/1.000 rating, nên lệnh chạy mặc định dùng mức sau.

`evaluate()` gọi `predict()` cho nhóm rating nếu `supports_ratings=True`, và gọi `recommend(user,k)` cho từng user trong nhóm xếp hạng. Sách liên quan để chấm Top-K là **rating test từ 7 trở lên**, không gồm tương tác 0 dù LightFM dùng rating 0 làm nhãn dương khi học.

Precision/Recall tổng được lấy trung bình theo user: mỗi user có trọng số ngang nhau. Sách test chưa từng có trong train vẫn được thống kê và vẫn nằm trong mẫu số Recall nếu liên quan.

Các chỉ số phụ cần hiểu:

| Cột | Ý nghĩa |
|---|---|
| `rating_coverage` | Tỷ lệ dòng rating trong nhóm đánh giá có dự đoán hữu hạn hợp lệ |
| `cold_user`, `cold_item` | User/sách không có rating tường minh train; không nhất thiết chưa có tương tác 0 |
| `rating_cold_users`, `rating_cold_items` | Số dòng nhóm rating mang các cờ cold tương ứng, không phải số ID duy nhất |
| `warm_mae`, `warm_rmse` | Sai số chỉ trên cặp cả user và sách đều có rating tường minh train |
| `recommendation_fill_rate` | Tổng sách trả về / tổng sách được yêu cầu trên nhóm user |
| `test_unknown_item_rows` | Số dòng test có sách chưa có bất kỳ tương tác train nào |

Coverage 100% có thể nhờ baseline cold-start, không chứng minh mọi dự đoán đều từ láng giềng/nhân tố học được. Hàm trả ba đối tượng: chỉ số tổng hợp, bảng dự đoán từng rating, bảng đánh giá xếp hạng từng user.

## 9. `experiments/`: nơi nối mọi bộ phận lại

### `check_data.py`

`source_hashes()` đọc ba CSV dạng byte và tính SHA-256. Có thể hiểu hash là dấu kiểm tra nội dung: so trước/sau để biết file có thay đổi không.

`main()` nạp, làm sạch, chia dữ liệu hai lần với cùng seed; kiểm tra không trùng cặp train/test, đủ tổng dòng, user test có trong train, rating đúng miền và phép chia lặp lại được. Sau đó ghi `results/data_validation.json` và `.csv`. Lệnh này kiểm tra dữ liệu, không huấn luyện 6 mô hình.

### `runner.py` — bộ điều phối chính

Đây là file nên đọc khi muốn biết toàn chương trình gọi nhau ra sao.

| Thành phần | Nhiệm vụ |
|---|---|
| `MODEL_NAMES` | Tên 6 mô hình hợp lệ |
| `create_model(name, config)` | Import đúng lớp, đưa tham số vào hàm khởi tạo, trả mô hình chưa fit |
| `prepare(config, data_dir)` | Đọc, làm sạch, chia tập, thu nhỏ quick, chọn nhóm đánh giá |
| `environment()` | Ghi phiên bản Python/thư viện và đường dẫn interpreter |
| `write_json()` | Lưu dữ liệu báo cáo thành JSON UTF-8 |
| `run_one()` | Fit một mô hình, đánh giá, tạo gợi ý mẫu, ghi file hoặc ghi lỗi |
| `cli()` | Đọc tham số terminal, tạo thư mục lượt chạy và điều phối toàn bộ |

**Quick thu nhỏ ra sao?** `prepare()` vẫn đọc/làm sạch và chia toàn bộ dữ liệu trước. Sau đó chỉ nhìn train để chọn tối đa 1.500 user nhiều tương tác nhất; trong số đó chọn tối đa 5.000 sách nhiều tương tác nhất. Train/test/catalog được lọc theo phạm vi này. Quick không phải lấy ngẫu nhiên vài dòng đầu và có thiên lệch về user hoạt động, sách phổ biến.

**Full khác gì?** Full giữ toàn catalog và toàn train/test đã làm sạch, không dùng giới hạn max user/item của quick. Tuy nhiên giới hạn **đánh giá** 50 user/1.000 rating vẫn còn nếu không đổi. Full training không đồng nghĩa đã chấm toàn test.

`run_one()` lấy mô hình từ `create_model()`, gọi `model.fit(books, train)` đúng một lần rồi gọi `evaluate()`. Sau đó nó chọn một user mẫu, gọi `recommend()` và lưu các bảng. Nếu truyền `--book-id` cho Content-Based, nó gọi thêm `similar_books()`.

`--user-id` chỉ đổi user dùng cho danh sách gợi ý mẫu. Nó không biến toàn bộ phép đánh giá thành đánh giá một user đó. Khi không truyền, user mẫu thường là user đầu trong nhóm đánh giá.

`cli()` tạo thư mục `quick_<giờ UTC>/` hoặc `full_<giờ UTC>/`, chuẩn bị dữ liệu chung một lần và lần lượt chạy các mô hình. Kết quả được lưu sau mỗi mô hình; một mô hình lỗi Python thông thường thì các mô hình sau vẫn được thử. Kết thúc, nó kiểm tra hash dữ liệu và cập nhật bảng so sánh ở gốc `results/`.

Lệnh chạy riêng một mô hình thất bại sẽ thoát với mã 1. `run_all` có thể kết thúc với `completed_with_errors`, nên cần xem cột `status` của từng mô hình, không chỉ thấy terminal chạy xong là coi tất cả thành công. Native crash của thư viện có thể dừng cả tiến trình, không được bảo đảm bắt bằng `except Exception`.

### Các file `run_*.py`

Các file này rất ngắn vì chỉ là cổng gọi cùng bộ điều phối:

| File | Lệnh gọi bên trong |
|---|---|
| `run_all.py` | `cli()` — mặc định cả 6 mô hình hoặc danh sách `--models` |
| `run_popularity.py` | `cli('popularity')` |
| `run_content_based.py` | `cli('content_based')` |
| `run_user_cf.py` | `cli('user_cf')` |
| `run_item_cf.py` | `cli('item_cf')` |
| `run_svd.py` | `cli('svd')` |
| `run_lightfm.py` | `cli('lightfm')` |

Thuật toán nằm trong `src/models/`, không nằm trong các file chạy này.

### Luồng gọi đầy đủ của một lệnh

```text
python run.py -m experiments.run_all --mode quick
  → run.py: main()
    → Python của môi trường đầy đủ chạy experiments/run_all.py
      → runner.cli()
        → đọc tham số và tạo ExperimentConfig
        → tính hash CSV, tạo thư mục kết quả
        → runner.prepare()
          → load_data() → load_csv() cho Books, Ratings, Users
          → preprocess()
          → split_interactions()
          → chọn phạm vi quick
          → evaluation_cohorts()
        → với mỗi tên mô hình: runner.run_one()
          → create_model()
          → model.fit(books, train)
          → evaluate()
            → model.predict() nếu mô hình hỗ trợ rating
            → model.recommend() → model.score_items() → _result()
          → model.recommend() cho user mẫu
          → ghi metrics, predictions, ranking, recommendations
        → kiểm tra hash CSV, ghi bảng so sánh và manifest
```

Ví dụ với Item-CF, `create_model('item_cf', config)` tạo `ItemBasedCF(neighbors=30)`. `fit()` được kế thừa từ `SparseCF`, tạo ma trận. `recommend()` được kế thừa từ `BaseRecommender`, nhưng khi nó gọi `score_items()` thì Python chạy phần trong `item_based_cf.py`.

## 10. `results/`: đọc kết quả theo thứ tự nào?

### Các file ở ngay dưới `results/`

| File | Nội dung |
|---|---|
| `data_validation.csv` | Báo cáo chất lượng/số lượng dữ liệu dạng bảng |
| `data_validation.json` | Cùng báo cáo nhưng cấu trúc lồng nhau dễ giữ các nhóm thông tin |
| `model_comparison.csv` | Bảng tổng hợp các mô hình trong lượt thực nghiệm được cập nhật gần nhất |
| `model_comparison.json` | Thêm đường dẫn run, manifest và danh sách kết quả mô hình |

CSV dễ mở như bảng tính. JSON là định dạng dữ liệu dùng cặp tên–giá trị, tiện cho chương trình và thông tin nhiều cấp. Chúng là báo cáo, không phải đầu vào huấn luyện.

### Bên trong `quick_20261018T085504340706Z/`

Tên thư mục là một dấu thời gian UTC ngày 18/10/2026. Mỗi lần chạy mới sẽ tạo thư mục khác. `manifest.json` ghi cấu hình, môi trường, hash CSV, phạm vi train/test và trạng thái; `model_comparison.csv/.json` giữ kết quả của riêng lượt đó, không đổi theo lượt sau.

Mỗi mô hình có đủ năm file, đặt theo quy luật sau:

| Tiền tố | Các file của mô hình |
|---|---|
| `popularity` | `popularity_metrics.csv`, `popularity_metrics.json`, `popularity_predictions.csv`, `popularity_ranking.csv`, `popularity_recommendations.csv` |
| `content_based` | `content_based_metrics.csv`, `content_based_metrics.json`, `content_based_predictions.csv`, `content_based_ranking.csv`, `content_based_recommendations.csv` |
| `user_cf` | `user_cf_metrics.csv`, `user_cf_metrics.json`, `user_cf_predictions.csv`, `user_cf_ranking.csv`, `user_cf_recommendations.csv` |
| `item_cf` | `item_cf_metrics.csv`, `item_cf_metrics.json`, `item_cf_predictions.csv`, `item_cf_ranking.csv`, `item_cf_recommendations.csv` |
| `svd` | `svd_metrics.csv`, `svd_metrics.json`, `svd_predictions.csv`, `svd_ranking.csv`, `svd_recommendations.csv` |
| `lightfm` | `lightfm_metrics.csv`, `lightfm_metrics.json`, `lightfm_predictions.csv`, `lightfm_ranking.csv`, `lightfm_recommendations.csv` |

**`*_metrics.csv/.json`:** chỉ số tổng hợp một mô hình, gồm status, backend, MAE/RMSE nếu có, Precision/Recall, phạm vi đánh giá và thời gian. Hai định dạng lưu cùng hàng chỉ số.

**`*_predictions.csv`:** mỗi dòng là một rating test được chọn, có `User-ID`, `Book-ID`, `Book-Rating` thực tế, cờ `cold_user`, `cold_item` và `prediction`. Với Popularity/Content-Based/LightFM, prediction trống vì chúng không dự đoán rating trong API dự án. File vẫn được ghi để giữ thông tin nhóm đánh giá chung.

**`*_ranking.csv`:** mỗi dòng là một user được đánh giá, có `precision_at_k`, `recall_at_k`, `relevant_count` (số sách thích trong test), `returned_count` (số sách trả về), `unknown_train_relevant` (số sách liên quan chưa có tương tác train). File này không liệt kê sách đề xuất.

**`*_recommendations.csv`:** danh sách sách thực sự đề xuất cho **một user mẫu**, không phải toàn bộ user đánh giá. Có mã sách, tên, tác giả, số rating tường minh train và score. User mẫu được in ở terminal, không có cột user trong bảng này; muốn biết chắc user nào nên truyền `--user-id`.

Một số file có thể được tạo thêm trong lượt khác: `content_based_similar_books.csv` khi yêu cầu sách tương tự, `<model>_error.txt` khi xảy ra lỗi. Chúng không bắt buộc có trong lượt thành công hiện tại.

**Thứ tự đọc hữu ích:** xem `status` trong `model_comparison.csv` → chọn chỉ số đúng mục tiêu → mở `*_recommendations.csv` xem gợi ý → mở predictions/ranking để tìm hiểu lỗi chi tiết → xem manifest nếu cần biết cấu hình.

### Kết quả đã lưu của lượt đủ 6 mô hình

| Mô hình | MAE | RMSE | Precision@10 | Recall@10 |
|---|---:|---:|---:|---:|
| Popularity | Không áp dụng | Không áp dụng | 0,004 | 0,010 |
| Content-Based | Không áp dụng | Không áp dụng | 0,036 | 0,12075 |
| User-CF | 1,5934 | 2,0586 | 0,006 | 0,00682 |
| Item-CF | 1,2949 | 1,7703 | 0 | 0 |
| SVD | 1,1936 | 1,6079 | 0,004 | 0,00583 |
| LightFM logistic | Không áp dụng | Không áp dụng | 0,006 | 0,012 |

Lượt này dùng catalog 5.000 sách, 137.953 tương tác train, 32.338 tương tác test; chỉ lấy 1.000 rating và 50 user để đánh giá. Content-Based đứng đầu Top-10 và SVD có sai số rating thấp nhất trong lượt này. Precision 0,036 nghĩa là trung bình 3,6% vị trí đề xuất trúng tập liên quan đã biết; không phải 36% hay “độ chính xác toàn hệ thống”.

Item-CF được 0 ở Top-10 nghĩa là không có lượt trúng trong nhóm user được chọn, không có nghĩa chương trình không chạy hoặc mọi dự đoán rating đều sai. Không quan sát user chấm một sách cũng chưa đủ kết luận user ghét sách đó. Các con số này là đánh giá offline trên mẫu quick, chưa đại diện chất lượng khi dùng thực tế.

## 11. `tests/`: kiểm tra tính đúng của chương trình

### `test_metrics.py`

Ba test kiểm tra công thức: bỏ rating 0/giá trị không hữu hạn; trả kết quả thiếu khi không có dữ liệu chấm; xử lý danh sách ngắn và sách trùng khi tính Precision/Recall.

### `test_real_models.py`

`setUpClass()` dùng CSV thật, chuẩn bị một phạm vi nhỏ: tối đa 60 user, 400 sách, 3 user đánh giá, 20 rating, 2 epoch và 8 nhân tố. Năm test kiểm tra:

- Các mô hình loại lịch sử train, không trả sách trùng, score hữu hạn; xử lý user lạ và nhất quán giữa predict/score.
- LightFM fit trên dữ liệu thật; test này có thể bị skip nếu chưa cài thư viện.
- Content-Based tìm sách tương tự và báo lỗi với mã sách không có.
- Backend SVD NumPy tạo điểm hợp lệ, predict và score khớp nhau.
- Các mô hình phù hợp xử lý train chỉ có tương tác ngầm; catalog rỗng phải báo lỗi.

Tổng hiện có 8 test. Bộ test kiểm tra hành vi cần thiết, không chứng minh mọi trường hợp đều đúng hoặc chất lượng khuyến nghị đã tốt. Test không phải nơi tạo bảng so sánh thực nghiệm chính.

## 12. Các thư mục tự sinh: không cần đọc để hiểu thuật toán

| Thư mục/file | Vai trò |
|---|---|
| `.venv/Include`, `.venv/Lib`, `.venv/Scripts`, `pyvenv.cfg`, `.venv/.gitignore` | Cấu trúc môi trường Python cũ: thư viện, chương trình và cấu hình; không phải code thuật toán của dự án |
| `.cache/micromamba/` | Công cụ cài môi trường và gói tải về |
| `.cache/mamba-root/`, `.cache/packages/` | Dữ liệu đệm của quá trình thiết lập/thử môi trường |
| `.cache/surprise/` | Vị trí cache dành cho Surprise |
| `.cache/lightfm_real_train.npz` | Tệp ma trận đã lưu trong quá trình kiểm tra LightFM; luồng runner hiện tại không đọc tệp này để huấn luyện |
| `__pycache__/` và `*.pyc` | Bytecode Python tự tạo để hỗ trợ nạp module; có thể xuất hiện ở gốc và trong các gói |

Không cần sửa file nội bộ thư viện để đổi thuật toán dự án. Mã mình cần đọc nằm chủ yếu trong `src/` và `experiments/`.

## 13. Anh muốn thử thì chạy gì và mở gì?

Mở terminal ở thư mục gốc dự án. Các lệnh dưới đây dùng môi trường đầy đủ đã cài; nếu chuyển máy mới thì làm bước setup theo README trước.

**Chạy cả 6 mô hình theo cấu hình quick mặc định:**

```powershell
python run.py -m experiments.run_all --mode quick
```

**Chỉ thử Content-Based và lấy danh sách cho một user cụ thể:**

```powershell
python run.py -m experiments.run_content_based --user-id 17950
```

Nếu user đó không có hồ sơ thích trong phạm vi train đang xét, chương trình sẽ dùng Popularity. Vì thế chạy không lỗi chưa chắc có nghĩa đang tạo gợi ý cá nhân hóa cho user đó.

**Thử nhanh hơn với dữ liệu nhỏ:**

```powershell
python run.py -m experiments.run_all --mode quick --max-users 100 --max-items 1000 --eval-users 10 --eval-ratings 100 --epochs 5
```

Giảm các giới hạn sẽ đổi dữ liệu và kết quả; không đem con số đó so trực tiếp với run mặc định như thể cùng thí nghiệm.

**Chỉ chạy một số mô hình hoặc chọn backend SVD:**

```powershell
python run.py -m experiments.run_all --models popularity content_based svd
python run.py -m experiments.run_svd --svd-backend numpy
```

**Tìm sách giống một sách có trong catalog:**

```powershell
python run.py -m experiments.run_content_based --mode full --book-id 0195153448 --eval-users 10
```

Lệnh này dùng full để mã sách được tìm trong toàn catalog; nó vẫn fit và đánh giá mô hình theo quy trình runner, không phải chỉ tra cứu tức thì một mô hình đã lưu.

**Kiểm tra dữ liệu và chạy kiểm thử:**

```powershell
python run.py -m experiments.check_data
python run.py -m unittest discover -v
```

**Chạy train toàn bộ dữ liệu:**

```powershell
python run.py -m experiments.run_all --mode full
```

Đánh giá vẫn giới hạn mặc định. Muốn bỏ giới hạn đánh giá, thêm `--eval-users 0 --eval-ratings 0`; khối lượng tính sẽ lớn hơn nhiều, nhất là CF trên toàn catalog.

## 14. Lộ trình đọc code cho lần đầu

1. Đọc `src/utils/config.py` để biết tên tham số và đường dẫn.
2. Đọc `load_data.py`, rồi `preprocessing.py` để hiểu mô hình nhận dữ liệu nào.
3. Đọc `base.py`, rồi `popularity.py` để thấy một hệ gợi ý tối giản hoạt động.
4. Đọc `content_based.py` để hiểu cách biến văn bản thành số và xếp hạng.
5. Đọc `cf_common.py`, `user_based_cf.py`, `item_based_cf.py` để hiểu hai hướng lọc cộng tác.
6. Đọc `svd.py`, rồi `lightfm_model.py` để hiểu mô hình học vector.
7. Đọc `metrics.py`, `evaluate.py` để hiểu các con số chất lượng.
8. Đọc `runner.py` và `run.py` để nối luồng chạy; mở bảng kết quả bên cạnh.

Nếu muốn sửa quy tắc tính điểm, tìm trong model tương ứng. Muốn đổi dữ liệu được chọn, tìm `prepare()` trong runner. Muốn đổi cách chấm chất lượng, xem evaluation. Muốn đổi các giới hạn mặc định, xem config. Muốn xem hệ thống đang gợi ý sách gì, mở `*_recommendations.csv` của lượt vừa chạy.
