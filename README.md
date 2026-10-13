# Soru-Cevap Sistemlerinde Retrieval-Augmented Generation Yaklaşımlarının Karşılaştırmalı Değerlendirilmesi

Bu proje, farklı Retrieval-Augmented Generation (RAG) yapılandırmalarının belge tabanlı soru-cevap görevinde nasıl performans gösterdiğini karşılaştırır. Karşılaştırmayı yapabilen bir soru-cevap prototipi de geliştirilecektir. Hazır dil modelleri kullanılır, sıfırdan model eğitilmez.

## Neden bu proje?

Büyük dil modelleri kendilerine belge verilmediğinde kaynağı belirsiz ya da belgeyle desteklenmeyen yanıtlar üretebilir (halüsinasyon). Belgeler uzun olduğunda ve aynı kavram farklı biçimlerde ifade edildiğinde anahtar kelime araması da ilgili bilgiyi bulmakta zorlanır.

RAG bu iki sorunu birlikte ele alır:

1. Belgeler parçalara bölünür ve her parça bir vektöre (embedding) çevrilir.
2. Bir soru geldiğinde soruyla anlamca en yakın parçalar bulunur (retrieval).
3. Bu parçalar soruyla birlikte dil modeline verilir ve model kaynağa dayanan bir yanıt üretir.

Bu akışın her adımında birden fazla seçenek vardır: hangi embedding modeli, hangi arama yöntemi, hangi vektör altyapısı, hangi dil modeli. Bu proje, bu seçimlerin sonucu nasıl değiştirdiğini kontrollü deneylerle inceler.

## Cevap aradığımız sorular

**Ana soru:** Farklı RAG bileşenleri, farklı konulardaki Türkçe ve İngilizce belgelerde yanıt kalitesini, erişim başarısını ve işlem süresini nasıl etkiler?

Bu soruyu şu alt sorulara ayırıyoruz:

1. **RAG fark yaratıyor mu?** Aynı soru belgeden getirilen parçalarla ve parçasız sorulduğunda yanıtların doğruluğu ve kaynakla tutarlılığı nasıl değişiyor? Halüsinasyon ne kadar azalıyor?
2. **Embedding modeli ne kadar önemli?** multilingual-e5 ve BGE-M3 modelleri doğru belge parçalarını bulma başarısı bakımından nasıl karşılaştırılıyor?
3. **Retrieval yöntemi sonucu değiştiriyor mu?** Yoğun vektör erişimi ile hibrit erişim (anahtar kelime ve vektör birlikte) arasında erişim ve yanıt kalitesi farkı var mı?
4. **Vektör altyapısı seçimi ne fark yaratıyor?** Chroma, Qdrant ve FAISS aynı parçalar, aynı embedding ve aynı top-k ile karşılaştırıldığında indeksleme süresi ve sorgu süresi nasıl farklılaşıyor?
5. **Dil modeli seçimi yanıtı nasıl etkiliyor?** Farklı hazır dil modelleri aynı getirilen parçalarla hangi kalitede ve hangi hızda yanıt veriyor?
6. **Türkçe ile İngilizce arasında fark var mı?** Aynı içeriğe dayanan eşleştirilmiş Türkçe ve İngilizce sorularda performans farkı ne kadar ve bu farkın kaynağı ne (embedding, dil modeli, çeviri kalitesi)?
7. **Sistem ne zaman yanılıyor?** Hatalar retrieval aşamasından (yanlış parça getirildi) mı, yoksa üretim aşamasından (doğru parça verildi ama model yanlış yanıtladı) mı kaynaklanıyor?

## Yaklaşım

Tüm olası kombinasyonları denemek yerine temel bir yapılandırma sabit tutulur ve bileşenler tek tek değiştirilerek etkileri ayrı ayrı ölçülür. Ardından seçilmiş birkaç kombinasyon birbiriyle karşılaştırılır.

Türkçe ve İngilizce deneyleri, konu ve içerik farklarının dil etkisiyle karışmaması için aynı belge içeriği üzerinden yürütülür. Çeviri kaynaklı sınırlılıklar raporlanır.

## Nasıl değerlendiriyoruz?

- **RAGAS metrikleri:** Faithfulness (yanıt getirilen kaynaklarla tutarlı mı), Answer Relevancy (yanıt soruyla ilgili mi), Context Precision ve Context Recall (doğru parçalar bulunmuş mu)
- **Yanıt süresi:** Erişim ve üretim süreleri ayrı ayrı ölçülür
- **Referans yanıtlar ve insan kontrolü:** Otomatik değerlendirme tek doğruluk kanıtı sayılmaz. İki kişi seçilmiş yanıtları bağımsız inceler, uyuşmazlıklar kayıt altına alınır.

## Geliştirilecek sistem

Karşılaştırmalar bir web uygulamasında yapılabilecek:

- Kullanıcı bir belge yükler ve belge hakkında soru sorar
- Farklı RAG yapılandırmaları arasından seçim yapar
- Aynı soru için farklı yapılandırmaların yanıtlarını, kaynak parçalarını ve yanıt sürelerini yan yana görür
- Araştırmacı olarak toplu deney başlatır ve sonuçları CSV olarak dışa aktarır

Teknolojiler: Python, Streamlit (ön yüz), FastAPI (arka yüz), SQLite (deney kayıtları), pandas (sonuç analizi).

## Kapsam

- Yeni bir temel model veya algoritma geliştirildiği iddia edilmez. Katkı, deney düzeni, karşılaştırmalı bulgular ve hata analizidir.
- Deneylerde yalnızca kamuya açık ve kullanımı serbest belgeler kullanılır.
- OCR ve ek dosya biçimleri kapsam dışıdır, süre kalırsa ele alınır.

## Kaynaklar

1. Lewis ve diğerleri (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Gao ve diğerleri (2023). Retrieval-Augmented Generation for Large Language Models: A Survey. https://arxiv.org/abs/2312.10997
3. Karpukhin ve diğerleri (2020). Dense Passage Retrieval for Open-Domain Question Answering. https://aclanthology.org/2020.emnlp-main.550/
4. Es ve diğerleri (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. https://arxiv.org/abs/2309.15217
5. Wang ve diğerleri (2024). Multilingual E5 Text Embeddings: A Technical Report. https://arxiv.org/abs/2402.05672
6. Chen ve diğerleri (2024). BGE M3-Embedding. https://arxiv.org/abs/2402.03216
7. Douze ve diğerleri (2024). The Faiss library. https://arxiv.org/abs/2401.08281
