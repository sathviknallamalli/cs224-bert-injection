# cs224-bert-injection
Aakriti Lakshmanan, Aditya Tadimeti, Sathvik Nallamalli
<br>
The interpretability of Large Language Models (LLMS) like BERT (Devlin et al.,
2018) remains limited. Particularly, questions remain regarding how they generate
predictions for a masked word when given a syntactically ambiguous sentence.
Different replacements for the masked word can alter the sentence’s meaning,
thereby making BERT’s predictions diverse. Previous research in Hewitt and
Manning (2019) indicates LLMs can encode dependency parse tree information of
a sentence within the hidden vector representations of each word. Hewitt’s research
focused on the development of a structural probe matrix, which performs a linear
transformation from the squared L2 distances of the hidden vector representations to
the distance between words in the parse tree. Our research extends this to determine
whether BERT can use dependency parse tree information in its predictions. This
would indicate BERT can leverage syntactical rules for its outputs, yielding insights
for its predictions. To answer this, we perform an ‘injection’ on BERT by using the
structural probe matrix from Hewitt and Manning (2019) to apply a transformation
on BERT’s hidden vectors, allowing us to ‘push’ them towards towards a ‘gold’
dependency parse tree. After this ‘injection’, we pass the transformed hidden
vectors through the remaining layers of BERT and analyze the output probability
distribution for the masked word. We conducted a qualitative analysis, doing a
subjective review of the types of words the different injected models predicted
for the mask, as well as a quantitative analysis, via objective, proportion-based
metrics we devised using specific types of sentences. Our research supports the
hypothesis that BERT can use information encoded in dependency parse trees to
generate predictions that align with the added information
