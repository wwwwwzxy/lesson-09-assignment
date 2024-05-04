import jieba
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

# 文章文本
text = "迈向充满希望的新世纪——一九九八年新年讲话（附图片１张）中共中央总书记、国家主席江泽民（一九九七年十二月三十一日）１９９８年新年讲话《迈向充满希望的新世纪》发表。同胞们、朋友们、女士们、先生们：在１９９８年来临之际，我十分高兴地通过[中央人民广播电台]、[中国国际广播电台]和[中央电视台]，向全国各族人民，向[香港特别行政区]同胞、澳门和台湾同胞、海外侨胞，向世界各国的朋友们，致以诚挚的问候和良好的祝愿！……"

# 使用jieba进行分词
words = jieba.lcut(text)

# 使用TF-IDF计算权重
vectorizer = TfidfVectorizer(use_idf=True)
tfidf_matrix = vectorizer.fit_transform(words)
feature_names = vectorizer.get_feature_names_out()

# 获取每个词的TF-IDF值
tfidf_feature = {}
for word, tfidf in zip(feature_names, tfidf_matrix[0]):
    tfidf_feature[word] = tfidf

# 根据TF-IDF值排序并取前五个关键词
sorted_keywords = sorted(tfidf_feature.items(), key=lambda x: x[1], reverse=True)[:5]

# 输出关键词
for keyword, _ in sorted_keywords:
     print(keyword)