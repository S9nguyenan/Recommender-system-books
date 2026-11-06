# Book Recommender System

Dự án môn học Python chạy trong terminal: thực nghiệm sáu thuật toán khuyến
nghị sách trên Book-Crossing, so sánh dự đoán điểm và khuyến nghị Top-N.
Tái sử dụng phần đọc, làm sạch và chia dữ liệu; không sửa CSV gốc.

**Cập nhật:** đã cài đủ thư viện và chạy thành công cả sáu mô hình trên dữ
liệu thật ở chế độ quick. Môi trường đầy đủ là Python 3.11 tại
`%LOCALAPPDATA%\BookRecommender\env311`, dùng LightFM logistic trên Windows.
Chạy `python run.py -m experiments.run_all --mode quick`.

## Chạy từ thư mục gốc trong terminal VS Code

```powershell
python setup_environment.py 
python run.py -m experiments.check_data
python run.py -m experiments.run_all --mode quick
```

Môi trường đầy đủ đã được cài trên máy này, lệnh setup có thể bỏ qua.
`run.py` tự gọi đúng Python và thiết lập PATH cho DLL Conda ở tiến trình con;
không cần activate hay thay Python hệ thống. Nếu chỉ dùng năm mô hình cốt lõi,
có thể tạo venv thông thường (cách này không giải quyết LightFM trên Python 3.14):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.check_data
```

Lệnh ghi đè hai báo cáo kiểm tra `results/data_validation.json` và
`results/data_validation.csv`. Đây là báo cáo dữ liệu, không phải kết quả mô hình.
CSV gốc không bị sửa; SHA-256 được kiểm tra trước/sau chạy.
Có thể chỉ định `--data-dir` và `--results-dir` bằng đường dẫn tuyệt đối.
Đường dẫn mặc định luôn tính từ vị trí `src/utils/config.py`.

## Vai trò các file

- `src/utils/config.py`: đường dẫn gốc và cấu hình năm, tuổi, seed, tỷ lệ test.
- `src/data_processing/load_data.py`: `load_csv()` đọc CSV UTF-8 nghiêm ngặt,
  giữ mã sách dạng chuỗi; `load_data()` trả ba bảng trong `BookCrossingData`.
- `src/data_processing/preprocessing.py`: `preprocess()` trả `PreparedData`,
  gồm catalog, người dùng, tất cả tương tác, điểm tường minh, tương tác ngầm
  và báo cáo. `split_interactions()` chia tương tác theo từng người dùng.
- `experiments/check_data.py`: chạy kiểm tra toàn bộ dữ liệu thật, kiểm tra
  các bất biến train/test và lưu báo cáo.

## Quy tắc xử lý

- Tên cột thực tế là `Book-ID`. Giữ số 0 đầu; không tự đổi hoa/thường mã sách.
- Chuẩn hóa khoảng trắng, chuỗi rỗng thành thiếu. Chỉ loại bản ghi trùng hoàn
  toàn; nếu một khóa có nội dung xung đột thì báo lỗi, không tùy tiện chọn dòng.
- Loại khóa người dùng không nguyên dương, khóa sách thiếu, điểm không nguyên
  hoặc ngoài 0–10. Loại tương tác không thuộc catalog/người dùng hợp lệ để
  các mô hình sau dùng cùng tập sách; thống kê số dòng bị loại.
- `0` chỉ là tương tác ngầm; `1–10` là điểm tường minh. Không điền điểm thiếu
  bằng 0, không đưa 0 vào RMSE/MAE về sau.
- Năm ngoài 1450–2026 hoặc không nguyên trở thành thiếu; tuổi ngoài 5–100
  hoặc không nguyên trở thành thiếu. Đây là chính sách cấu hình, không khẳng
  định mọi tuổi ngoài khoảng đều sai. Không loại người dùng vì thiếu tuổi.
- Với metadata lệch trường, giữ tên sách gốc, gắn `Metadata-Needs-Review`,
  bỏ metadata tác giả/nhà xuất bản/ảnh không đáng tin; không đoán nội dung
  để sửa. Tác giả/nhà xuất bản thiếu dùng `Unknown`.
- Chia tất cả tương tác cùng lúc với seed 42, tỷ lệ test 0.2 theo người dùng.
  Người có một tương tác ở train; người có ít nhất hai tương tác có tối thiểu
  một ở train và một ở test (lấy phần nguyên 20%, tối thiểu một ở test).
  Sau đó mới lấy điểm tường minh/tương tác ngầm trong train và test.
- Không có timestamp nên không thể kiểm thử theo thời gian. Người có tương tác
  trong train có thể chưa có điểm tường minh trong train; mô hình điểm số sau
  này cần xử lý cold start và báo coverage. Báo cáo số dòng test có sách chưa
  từng xuất hiện trong train, không âm thầm loại khỏi đánh giá.
- TF-IDF được fit trên metadata catalog đã biết trước (không dùng điểm test).
  Các thống kê điểm, tương đồng CF, hồ sơ sở thích và nhân tố chỉ học từ train.

## Cấu trúc bổ sung

```text
src/models/
    base.py                 # Catalog, loại sách đã thấy, baseline điểm
    cf_common.py            # Ma trận sparse dùng chung cho hai CF
    popularity.py
    content_based.py
    user_based_cf.py
    item_based_cf.py
    svd.py
    lightfm_model.py
src/evaluation/
    metrics.py              # Công thức MAE/RMSE và Precision/Recall
    evaluate.py             # Cohort đánh giá chung và thống kê coverage
experiments/
    runner.py               # CLI, chuẩn bị một lần, huấn luyện/đánh giá/lưu
    run_popularity.py
    run_content_based.py
    run_user_cf.py
    run_item_cf.py
    run_svd.py
    run_lightfm.py
    run_all.py
tests/
    test_metrics.py
    test_real_models.py      # Kiểm tra mô hình bằng các dòng CSV thật
results/
    model_comparison.csv
    model_comparison.json
    quick_<UTC timestamp>/  # Kết quả riêng từng lần chạy, không đè lịch sử
```

## Thuật toán và API

Các mô hình đều gọi `fit(books, train_interactions)` và
`recommend(user_id, n=10, exclude_seen=True)`, trả DataFrame chứa mã sách,
tên, tác giả, số đánh giá tường minh train và điểm xếp hạng. Sách đã có bất
kỳ tương tác train nào (kể cả 0 hoặc điểm thấp) đều bị loại khỏi gợi ý.

| File / class | Cách hoạt động |
|---|---|
| `popularity.py` / `PopularityRecommender` | Trung bình Bayesian `(count * mean + 10 * global_mean) / (count + 10)` trên điểm tường minh train. |
| `content_based.py` / `ContentBasedRecommender` | TF-IDF tên sách, tác giả, nhà xuất bản; cosine với trung bình vector sách user chấm >= 7 trong train. `similar_books(book_id, n)` tìm sách giống một sách. |
| `user_based_cf.py` / `UserBasedCF` | Sparse user–item từ điểm 1–10, cosine giữa user; trung bình điểm có trọng số từ tối đa 30 láng giềng. |
| `item_based_cf.py` / `ItemBasedCF` | Cosine item; với từng ứng viên chỉ dùng tối đa 30 sách tương tự trong lịch sử tường minh của user. Tính theo block, không tạo ma trận item–item toàn bộ. |
| `svd.py` / `SVDRecommender` | Ưu tiên Surprise SVD: global mean + user/item bias + tích nhân tố. Nếu chưa import được Surprise, `auto` dùng NumPy mini-batch SGD trên rating quan sát; tên backend luôn ghi trong báo cáo. |
| `lightfm_model.py` / `LightFMRecommender` | Thư viện LightFM, logistic, identity features. Nhãn +1 cho tương tác 0 hoặc điểm >= ngưỡng; nhãn -1 cho điểm tường minh thấp hơn ngưỡng. Không dự đoán điểm 1–10. |

Hai CF và SVD hỗ trợ `predict(user_id, book_id)` trả điểm giới hạn 1–10.
CF thiếu láng giềng dùng trung bình item, rồi user, rồi global train; nếu
train không có điểm tường minh thì baseline là trung điểm 5.5. SVD
thiếu user/item dùng các bias đã học, không dùng nhân tố chưa học. Content
thiếu hồ sơ thích và LightFM thiếu user dùng popularity train. Popularity,
content và LightFM không có MAE/RMSE vì score của chúng không phải dự đoán rating.
LightFM không xếp hạng item chưa có positive train; chúng vẫn nằm trong mẫu
số Recall nếu liên quan trong test. Đây là hạn chế identity features.

## Môi trường đầy đủ đã kiểm chứng

`environment.yml` chỉ định bộ thư viện tương thích: Python 3.11,
NumPy 1.26, Pandas 2.x, SciPy, scikit-learn, Surprise 1.1.4, LightFM 1.17.
LightFM là bản gốc từ conda-forge, không thay bằng thuật toán khác.

```powershell
python setup_environment.py
python run.py -m unittest discover -v
```

Installer tải micromamba chính thức vào `.cache/micromamba/`, dùng gói
conda-forge đã biên dịch, tạo môi trường riêng tại
`%LOCALAPPDATA%\BookRecommender\env311`. Spec và package cache cũng đặt dưới
`%LOCALAPPDATA%\BookRecommender` để tránh lỗi đường dẫn có dấu của micromamba.
Nếu môi trường đã có, installer giữ nguyên. Muốn dùng VS Code trực tiếp,
chọn interpreter `C:\Users\S9\AppData\Local\BookRecommender\env311\python.exe`
trong **Python: Select Interpreter**. `run.py` luôn chọn môi trường đầy đủ,
kể cả khi terminal vẫn dùng Python 3.14.

LightFM báo không có OpenMP trên Windows và chạy một thread. Bản này bị
`access violation` với WARP/BPR; đã xác nhận logistic chạy thật và dùng làm
mặc định. Trên Windows code từ chối loss khác logistic trước khi gọi kernel
native. Có thể chọn `--lightfm-loss warp` hoặc `bpr` ở nền tảng khác nhưng
chưa kiểm chứng tại máy này. Logistic chỉ dùng nhãn từ train đã quan sát;
không lấy test làm negative, không sinh rating giả, ô chưa quan sát vẫn trống.
Điểm logistic là score xếp hạng, không phải rating 1–10.

## Lịch sử môi trường Python 3.14

Các thư viện cốt lõi nằm trong `requirements.txt`. Surprise và LightFM là
dependencies tùy chọn để lỗi build không chặn năm mô hình còn lại.
Môi trường kiểm chứng tại máy này là Python 3.14.7 trên Windows. Đã tạo
`.venv` dùng `--system-site-packages` để dùng lại NumPy/Pandas/SciPy/sklearn
có sẵn và cài riêng Surprise 1.1.5:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install scikit-surprise
```

Môi trường `.venv` cũ vẫn được giữ nguyên, nhưng không có LightFM.
Các lệnh chạy thực nghiệm bên dưới dùng `run.py` để chọn môi trường đầy đủ.

Thử cài toàn bộ dependencies tùy chọn:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

LightFM 1.17 không cài được bằng pip trong Python 3.14: build isolation lỗi
`AttributeError: 'dict' object has no attribute '__LIGHTFM_SETUP__'`.
Đã cài setuptools 84.0.0, wheel 0.48.0, Cython 3.3.0 vào `.venv` và thử lại:

```powershell
.\.venv\Scripts\python.exe -m pip install setuptools wheel Cython
.\.venv\Scripts\python.exe -m pip install --no-build-isolation lightfm
```

Vẫn lỗi tương tự khi tạo metadata. Những lần chạy cũ ghi `failed` được giữ
nguyên trong lịch sử. Môi trường conda-forge Python 3.11 ở trên đã khắc phục
việc cài đặt và chạy thành công LightFM logistic.

## Chạy từng mô hình và toàn bộ

```powershell
python run.py -m experiments.run_popularity
python run.py -m experiments.run_content_based
python run.py -m experiments.run_user_cf
python run.py -m experiments.run_item_cf
python run.py -m experiments.run_svd
python run.py -m experiments.run_lightfm
python run.py -m experiments.run_all --mode quick
```

Tìm sách tương tự (mã phải nằm trong catalog của chế độ đang chạy):

```powershell
python run.py -m experiments.run_content_based --mode full --book-id 0195153448 --eval-users 10
python run.py -m experiments.run_user_cf --user-id 17950
python run.py -m experiments.run_svd --svd-backend surprise
python run.py -m experiments.run_svd --svd-backend numpy
```

`auto` chỉ chuyển sang NumPy khi import Surprise không được; lỗi runtime
khác được báo rõ. Cache import Surprise đặt trong `.cache/surprise/`.

Chạy training trên toàn bộ dữ liệu đã làm sạch, giữ giới hạn đánh giá mặc định:

```powershell
python run.py -m experiments.run_all --mode full
```

Chạy cả đánh giá không giới hạn (có thể rất lâu trên CPU):

```powershell
python run.py -m experiments.run_all --mode full --eval-users 0 --eval-ratings 0
```

Quick nhỏ hơn để tự học/debug:

```powershell
python run.py -m experiments.run_all --mode quick --max-users 100 --max-items 1000 --eval-users 10 --eval-ratings 100 --epochs 5
```

Chọn một số mô hình, không chạy lại các mô hình khác:

```powershell
python run.py -m experiments.run_all --models popularity user_cf svd
```

`run_all` nạp dữ liệu/chia tập một lần, tạo mô hình độc lập, tiếp tục nếu một
mô hình lỗi. Mỗi mô hình chỉ fit một lần trong một lượt chạy. Không tự tái
sử dụng cache mô hình cũ vì có nguy cơ khác cấu hình/dữ liệu.
Lệnh mô hình riêng thoát mã 1 nếu thất bại; `run_all` hoàn thành cả khi có
mô hình thất bại, cần đọc `status` từng dòng và `manifest.status`.

## Phạm vi dữ liệu, chỉ số và tính công bằng

- Seed mặc định 42; mọi mô hình dùng cùng train, test, catalog và cohort đánh
  giá trong một run. `quick` chọn tối đa 1.500 user nhiều tương tác train,
  rồi 5.000 item phổ biến trong train của các user này; giữ test thuộc phạm
  vi đó. Đây là tập con thiên về user hoạt động và sách phổ biến, không đại
  diện toàn bộ dữ liệu. Số test bị loại khỏi quick được ghi trong manifest.
- `full` không giới hạn user/item train, dùng catalog đầy đủ, giữ cả item
  chưa thấy trong train ở test. `--max-users/--max-items` chỉ tác động quick.
- Mặc định đánh giá tối đa 1.000 rating tường minh test và 50 user có ít nhất
  một sách liên quan trong test, lấy mẫu cùng seed cho tất cả mô hình.
  `--eval-users 0 --eval-ratings 0` bỏ hai giới hạn. Full training không tự
  có nghĩa là đã đánh giá toàn bộ test.
- MAE là trung bình sai số tuyệt đối; RMSE là căn trung bình bình phương sai
  số. Chỉ rating thật 1–10 và dự đoán hữu hạn mới được tính. Báo cáo số hàng
  đánh giá, coverage và số hàng cold user/item (dựa trên explicit train).
  `warm_mae/warm_rmse` dùng cùng tập con user và item có explicit train.
  MAE/RMSE tổng có tính baseline cold-start, không tuyên bố tất cả là dự
  đoán từ láng giềng hay nhân tố học được.
- Relevance là rating test >= 7, thay đổi qua `--relevance-threshold`.
  Rating test 0 không là relevance. Precision@K = số hit / K, kể cả khi mô
  hình trả ít hơn K; Recall@K = số hit / tổng sách liên quan của user trong
  test thuộc phạm vi run. Lấy trung bình macro theo user. Không có relevant
  test thì bỏ khỏi hai trung bình và ghi số user đó riêng.
- Candidate là toàn bộ catalog thuộc phạm vi run, trừ mọi sách user đã tương
  tác train. Không lấy một tập negative nhỏ để làm tăng giả kết quả. Những
  sách chưa được quan sát ở test được coi không phải hit; do thiếu nhãn,
  Precision/Recall offline không chứng minh chúng là sách user không thích.
- Sách cold-start liên quan vẫn nằm trong mẫu số Recall. Báo cáo từng user
  có `unknown_train_relevant`, số sách trả về, và `recommendation_fill_rate`.
  Chưa có chia theo thời gian hoặc tuning bằng validation; không chọn tham
  số tốt nhất theo test. Kết quả hiện tại là baseline thực nghiệm.

## Tài nguyên và tùy chọn

`ExperimentConfig` trong `config.py` giữ mặc định CLI. Có thể thay đổi
`--neighbors`, `--factors`, `--epochs`, `--k`, `--max-features`, `--seed`.
CF dùng sparse, cosine theo user hoặc block item 256 x lịch sử user; chỉ
giữ cache điểm cho tối đa tám user. Content không tạo ma trận toàn bộ sách
với nhau. SVD NumPy cập nhật từng mini-batch 256 rating quan sát.
LightFM dùng một thread để tái lập. Không yêu cầu GPU.

Full CF phải tính điểm cho hàng trăm nghìn ứng viên với mỗi user, có thể rất
chậm; nên giảm số user/rating đánh giá trước. Giới hạn quick thay đổi phạm
vi thực nghiệm và chất lượng, không chỉ thời gian chạy.

## File kết quả và kiểm tra

Mỗi thư mục run có `manifest.json` (cấu hình, phiên bản, SHA-256, phạm vi),
`model_comparison.csv/json` và các file từng mô hình:

- `<model>_metrics.csv/json`: trạng thái, backend, thời gian, chỉ số và coverage.
- `<model>_recommendations.csv`: ví dụ Top-N được in trong terminal.
- `<model>_predictions.csv`: điểm test trong cohort, dự đoán và cờ cold-start.
- `<model>_ranking.csv`: Precision/Recall, relevance và fill theo user.
- `<model>_error.txt`: traceback khi có lỗi. Model thất bại không có chỉ số giả.

Hai file `results/model_comparison.csv/json` trỏ tới kết quả lần chạy gần
nhất; JSON kèm manifest và đường dẫn run. Lịch sử timestamp giữ nguyên.

```powershell
python run.py -m compileall -q src experiments tests
python run.py -m experiments.check_data
python run.py -m unittest discover -v
```

Các kiểm tra mô hình dùng tập con CSV thật để kiểm tra loại sách đã xem,
fallback user/item lạ, tính nhất quán giữa `predict` và score, và tìm sách
tương tự. Các ví dụ số nhỏ trong test công thức chỉ xác minh toán học,
không được lưu như dữ liệu hay kết quả thực nghiệm.

## Kết quả đã chạy thực tế ngày 18/10/2026

Lệnh: `.venv\Scripts\python.exe -m experiments.run_all --mode quick`.
Run `results/quick_20261018T083527000182Z/`: 5.000 sách, 137.953 tương tác
train, 32.338 tương tác test. Có 166.263 dòng test ngoài quick scope,
không được xem là đã đánh giá. Cohort chung là 1.000 rating và 50 user,
K=10, relevance >=7. Cả ba mô hình dự đoán điểm đều đánh giá 1.000/1.000
rating (bao gồm baseline cold-start); có 5 dòng cold item theo explicit
train, không có cold user trong cohort. Warm metrics dùng 995 dòng.

| Mô hình | Trạng thái | MAE | RMSE | Precision@10 | Recall@10 |
|---|---|---:|---:|---:|---:|
| Popularity | Thành công | — | — | 0.004000 | 0.010000 |
| Content-Based | Thành công | — | — | 0.036000 | 0.120746 |
| User-CF | Thành công | 1.593439 | 2.058602 | 0.006000 | 0.006818 |
| Item-CF | Thành công | 1.294905 | 1.770331 | 0.000000 | 0.000000 |
| SVD (Surprise 1.1.5) | Thành công | 1.193552 | 1.607905 | 0.004000 | 0.005833 |
| LightFM | Lỗi cài thư viện | — | — | — | — |

Recall/Precision bằng 0 của Item-CF là kết quả thật trên 50 user này, không
phải chỉ số giả hoặc lỗi được thay bằng 0. Không suy rộng thứ hạng mô hình
từ một tập con nhỏ; các chỉ số đo hai mục tiêu khác nhau.

Đã chạy thêm Popularity với full training và catalog:

```powershell
python run.py -m experiments.run_popularity --mode full --eval-users 10 --eval-ratings 100 --results-dir results/full_validation
```

Catalog 271.359 sách, train 832.538 dòng, test 198.601 dòng, giữ nguyên
32.675 dòng test có item chưa xuất hiện trong train. Chỉ đánh giá Top-N
cho 10 user, không phải toàn bộ test. Xem `results/full_validation/`.
Chưa chạy full training cho Content/CF/SVD/LightFM. Bảng phía trên là lần
chạy lịch sử trên `.venv`; xem lần chạy đầy đủ sáu mô hình ở phần cập nhật.

Đã kiểm tra cú pháp/import, chạy lại `check_data` thành công. Unittest có
Lần kiểm tra trên `.venv` cũ có 8 test: 7 đạt, 1 skip do LightFM chưa cài.
Đã kiểm tra file kết
quả của năm mô hình thành công dùng đúng cùng cohort rating/user; dòng
LightFM thất bại không chứa metric. Các lần chạy lỗi trước khi sửa vẫn
được giữ trong thư mục timestamp cũ để không ghi đè lịch sử.
Ghi chú cài thư viện đã được gộp trong các phần môi trường Python ở README này.

## Cập nhật: sáu mô hình đều chạy thành công

Môi trường mới: Python 3.11.17, NumPy 1.26.4, Pandas 2.3.3, SciPy 1.17.1,
scikit-learn 1.9.1, Surprise 1.1.4 và LightFM 1.17.
Lệnh `python run.py -m experiments.run_all --mode quick` hoàn tất tại
`results/quick_20261018T085504340706Z/`, cùng phạm vi 5.000 sách, 137.953
tương tác train, cohort 1.000 rating / 50 user như bảng lịch sử phía trên.

| Mô hình | Trạng thái | MAE | RMSE | Precision@10 | Recall@10 |
|---|---|---:|---:|---:|---:|
| Popularity | Thành công | — | — | 0.004000 | 0.010000 |
| Content-Based | Thành công | — | — | 0.036000 | 0.120746 |
| User-CF | Thành công | 1.593439 | 2.058602 | 0.006000 | 0.006818 |
| Item-CF | Thành công | 1.294905 | 1.770331 | 0.000000 | 0.000000 |
| SVD (Surprise 1.1.4) | Thành công | 1.193552 | 1.607905 | 0.004000 | 0.005833 |
| LightFM (logistic) | Thành công | — | — | 0.006000 | 0.012000 |

8/8 unittest đạt, không skip. Kiểm tra dữ liệu chạy thành công trong môi
trường mới, hash ba CSV gốc không đổi. Các file `results/model_comparison.*`
đã trỏ tới run sáu mô hình thành công này. LightFM logistic được chọn sau
khi bản Windows crash với WARP/BPR; không coi logistic là kết quả WARP.
Chưa chạy full training/đánh giá không giới hạn cho cả sáu mô hình.

```powershell
python run.py -m experiments.run_lightfm
python run.py -m experiments.run_all --mode quick
python run.py -m experiments.run_all --mode full
```
