import torch
import sys
sys.path.append(r'./pretraining')
from simctg import SimCTGPretraining
# load an off-the-shelf Japanese GPT (https://huggingface.co/colorfulscoop/gpt2-small-ja)
model_path = r"./model/gpt2-small-jae/"
model = SimCTGPretraining(model_path)
model.eval()

'''
   Prepare text prefix input. The prefix is copied from a random Japanese Wikipedia 
   page here (https://ja.wikipedia.org/wiki/%E8%87%A5%E9%BE%8D%E6%A1%9C).
'''
#text = r'臥龍桜（がりゅうざくら）は、岐阜県高山市一之宮町にある一本桜。龍が地'
text = r"３月も中盤に差し掛かり、暖かくなりましたね。週末にもなると"
tokens = model.tokenizer.tokenize(text)
input_ids = model.tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.LongTensor(input_ids).view(1,-1)

# (1) use contrastive search to generate the result
beam_width, alpha, decoding_len = 5, 0.6, 128
eos_token = model.tokenizer.eos_token
print (model.fast_contrastive_search(input_ids, beam_width, alpha, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に染みつく様子を図案化したもので、樹齢400年
   を越す日本さくら名所100選に選定されている。一之宮町指定天然記念物。岐阜県飛騨地方(東濃地方)の山間地に生育し、約
   1万年前に絶滅したと考えられている。「花の本」とも称され、開花期は5月上旬から下旬までで、桜の枝張りは濃緑色である。
   花は直径約10cmの花弁を咲かせる八重咲きで、花弁の色は紅紫色で、雄しべは4本、雌しべは1本ある。雄しべの先
'''

# (2) use nucleus sampling to generate the result
nucleus_p, decoding_len = 0.95, 128
eos_token = model.tokenizer.eos_token
print (model.nucleus_sampling(input_ids, nucleus_p, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む奇岩に由来する。毎年5月上旬には多くの花見
   客が訪れている。かつて、雪見の藩お抱え家臣、雲口である長久城主長久竜泰が祭っている「月輪寺」には手水鉢が2つあり、長
   久氏の勢力が強まると同時に関連する寺もあり、山を挟むように吉野側の赤峰山から北へ順に樹齢250年を越してきたが、江戸時
   代に廃材が搬出されてから薪が取れなくなっている。古い株は毎年12月の初午に燃えつき風雨が吹き荒れて朽ち果てる。根は分枝
'''

# (3) use greedy search to generate the result
decoding_len = 128
eos_token = model.tokenizer.eos_token
print (model.greedy_search(input_ids, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中に棲む龍の棲むとされる桜で、樹齢は1000年以上。樹
   高は10mほどで、幹周りは8mほどになる。樹齢は300年ほどで、樹高は20mほどになる。樹形が整っており、枝張りも良く、樹勢も
   旺盛である。樹形は、樹高が1mほどで、幹周りは4mほどになる。枝張りはよく発達し、樹勢は旺盛である。冬になると、幹周りの
   樹冠が紅葉する。また、紅葉の時期には、樹冠が赤く紅葉する。樹
'''

# (4) use beam search to generate the result
beam_width, decoding_len = 10, 128
eos_token = model.tokenizer.eos_token
print (model.beam_search(input_ids, 10, decoding_len, eos_token))
'''
   臥龍桜(がりゅうざくら)は、岐阜県高山市一之宮町にある一本桜。龍が地中深くに咲く桜で、岐阜県の天然記念物に指定されている。
   岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮にある一本桜である。龍が地中深くに
   咲く桜で、岐阜県の天然記念物に指定されている。岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一
   之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山市一之宮町一之宮(いちのみやちょういちのみや)は、岐阜県高山
'''