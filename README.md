# Side-Projects

1. [CAPTCHA_OCR](https://github.com/ShSeanLee/Side-Projects/tree/master/CAPTCHA_OCR)
- Keras 제공 CAPTCHA OCR 라이브러리 활용
- Tensorflow, ONNX 형식으로 제공
- WOL 자동화 목적

2. [System_Trading](https://github.com/ShSeanLee/Side-Projects/tree/master/System_Trading)
- 한국투자증권 API 활용 시스템 트레이딩 구현
- 변동성 돌파전략(by 래리 윌리엄스) 활용
- 미국주식 IVV 자동매매 전략으로 변경 예정

3. [Movie_Recommendation](https://github.com/ShSeanLee/Side-Projects/tree/master/Movie_Recommendation)
- TMDB 활용 영화 추천 서비스 구현
    1. Demographic Filtering (인구통계학적 필터링)
        - 많은 사람이 높은 평점을 매긴 순으로
    1. Content Based Filtering (컨텐츠 기반 필터링)
        - 컨텐츠 기반 유사 아이템 추천(줄거리, 배우, 감독, 키워드)
        - NLP, Cosine유사도 활용
    1. Collaborative Filtering (협업 필터링)
        - 비슷한 영화 취향을 가진 사람끼리 매칭시켜서 추천 해줌
        - Surprise 활용
        - 리뷰에 기반하여 비슷한 취향을 가진 사람이 본 것 추천
- Cosine유사도, Surprise 활용 맞춤 추천 서비스