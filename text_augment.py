from calendar import c
import gzip
import shutil
import sqlite3
import pandas as pd
import random
from math import ceil
import MeCab
import ipadic
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TextAugment():
    def __init__(self, folder_path, file_name):
        self.text_file_path = folder_path + file_name


        with gzip.open(folder_path+'wnjpn.db.gz', 'rb') as f_in:
            with open(folder_path+'wnjpn.db', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        conn = sqlite3.connect(folder_path+"wnjpn.db")
        q = 'SELECT synset,lemma FROM sense,word USING (wordid) WHERE sense.lang="jpn"'
        self.sense_word = pd.read_sql(q, conn)
        self.stop_words = pd.read_csv(folder_path+"Japanese.txt",header=None)[0].to_list()  
    
    def get_synonyms(self, word):
        synsets = self.sense_word.loc[self.sense_word.lemma == word, "synset"]
        synset_words = set(self.sense_word.loc[self.sense_word.synset.isin(synsets), "lemma"])

        if word in synset_words:
            synset_words.remove(word)

        return list(synset_words)
    
    def wakati_text(self, text, hinshi=['名詞', '形容詞']):
        # Windowsの場合
        # m = MeCab.Tagger(ipadic.MECAB_ARGS)

        # Mac or Linuxの場合 -> Neologd
        m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
        
        p = m.parse(text)
        p_split = [i.split("\t") for i in p.split("\n")][:-2]

        # 原文の分かち書き
        raw_words = [x[0] for x in p_split]

        # 同義語検索用の単語の原型リスト（品詞を絞る）
        second_half = [x[1].split(",") for x in p_split]
        original_words = [x[6] if x[0] in hinshi else "" for x in second_half]
        original_words = ["" if word in self.stop_words else word for word in original_words]

        return raw_words, original_words

    def synonym_replacement(self, raw_words, original_words, n):
        new_words = raw_words.copy()

        # 同義語に置き換える単語をランダムに決める
        original_words_idx = [i for i, x in enumerate(original_words) if x != ""]
        random.shuffle(original_words_idx)

        # 指定の件数になるまで置き換え
        num_replaced = 0
        for idx in original_words_idx:
            raw_word = raw_words[idx]
            random_word = original_words[idx]
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == raw_word else word for word in new_words]
                #print(random_word, " → ", synonym)
                num_replaced += 1
            if num_replaced >= n:
                break
        #print(f"原文：{(''.join(raw_words))}")
        #print(f"変換後：{(''.join(new_words))}")
        #print("-"*50)

        return new_words

    def random_insertion(self, raw_words, original_words, n):
        new_words = raw_words.copy()
        for _ in range(n):
            new_words = self.add_word(new_words, original_words)
        #print(f"原文：{(''.join(raw_words))}")
        #print(f"変換後：{(''.join(new_words))}")
        #print("-"*50)
        return new_words

    def add_word(self, new_words, original_words):
        synonyms = []
        counter = 0
        insert_word_original = [x for x in original_words if x]
        if len(insert_word_original)==0 or new_words == None:
            pass
        else:
            while len(synonyms) < 1:
                random_word = insert_word_original[random.randint(0, len(insert_word_original)-1)]
                synonyms = self.get_synonyms(random_word)
                counter += 1
                if counter >= 10:
                    return
            random_synonym = synonyms[0]
            random_idx = random.randint(0, len(new_words)-1)
            #print(f"挿入する単語：{random_synonym}")
            new_words.insert(random_idx, random_synonym)
        return new_words
        
    def random_deletion(self, words, p):
        # 1文字しかなければ削除しない
        if len(words) <= 1:
            return words

        # 確率pでランダムに削除
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)
            else:
                #print(f"削除：{word}")
                pass

        # 全て削除してしまったら、ランダムに1つ単語を返す
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]

        #print(f"原文：{(''.join(words))}")
        #print(f"変換後：{(''.join(new_words))}")
        #print("-"*50)
        return new_words
    
    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            nwords = self.swap_word(new_words)

        #print(f"原文：{(''.join(words))}")
        #print(f"変換後：{(''.join(new_words))}")
        #print("-"*50)
        return new_words

    def swap_word(self, new_words):
        if len(new_words) == 0:
            pass
        else:
            random_idx_1 = random.randint(0, len(new_words)-1)
            random_idx_2 = random_idx_1
            counter = 0
            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, len(new_words)-1)
                counter += 1
                if counter > 3:
                    return new_words
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
            #print(new_words[random_idx_1], "⇔", new_words[random_idx_2])

        return new_words
    
    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

        # 分かち書き
        raw_words, original_words = self.wakati_text(sentence)
        num_words = len(raw_words)

        augmented_sentences = []
        techniques = ceil(alpha_sr) + ceil(alpha_ri) + ceil(alpha_rs) + ceil(p_rd)
        if techniques == 0:
            return

        num_new_per_technique = int(num_aug/techniques)+1

        #ランダムに単語を同義語でn個置き換える
        if (alpha_sr > 0):
            n_sr = max(1, int(alpha_sr*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.synonym_replacement(raw_words,original_words ,n_sr)
                if a_words:
                    augmented_sentences.append(''.join(a_words))

        #ランダムに文中に出現する単語の同義語をn個挿入
        if (alpha_ri > 0):
            n_ri = max(1, int(alpha_ri*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_insertion(raw_words,original_words, n_ri)
                if a_words:
                    augmented_sentences.append(''.join(a_words))

        #ランダムに単語の場所をn回入れ替える
        if (alpha_rs > 0):
            n_rs = max(1, int(alpha_rs*num_words))
            for _ in range(num_new_per_technique):
                a_words = self.random_swap(raw_words, n_rs)
                if a_words:
                    augmented_sentences.append(''.join(a_words))

        #ランダムに単語を確率pで削除する
        if (p_rd > 0):
            for _ in range(num_new_per_technique):
                a_words = self.random_deletion(raw_words, p_rd)
                if a_words:
                    augmented_sentences.append(''.join(a_words))

        #必要な文章の数だけランダムに抽出
        random.shuffle(augmented_sentences)
        augmented_sentences = augmented_sentences[:num_aug]

        #原文もリストに加える
        augmented_sentences.append(sentence)

        return augmented_sentences
    
    def regular_text(self):
        doc_list = pd.read_csv(self.text_file_path)["text"].values.tolist()
        regular_document = []

        print("start regular")
        for doc in tqdm(doc_list):
            if "その他" not in doc and re.search(r'(<@U.*>)', doc):
                texts = doc.split("\n")
                for text in texts:
                    text = re.sub(r'(<@U.*>)', '', text)
                    text = re.sub(r'(:[a-zA-Z0-9_-]+:)', '', text)
                    text = re.sub(r'(<https:.*>)', '', text)
                    text = re.sub(r'(<http:.*>)', '', text)
                    text = re.sub(r'(\n|\s)', '', text)
                    if len(text) > 15:
                        regular_document += [text]

            else:
                doc = re.sub(r'(\n|\s)', '', doc)
                text = doc.split("その他")[-1]
                text = re.sub(r'(:[a-zA-Z0-9_-]+:)', '', text)
                text = re.sub(r'(<https:.*>)', '', text)
                text = re.sub(r'(<http:.*>)', '', text)
                text = re.sub(r'(<@U.*>)', '', text)
                if len(text) > 15:
                        regular_document += [text]
        
        return regular_document

if __name__ == "__main__":
    folder_path = "./data/user_data/"
    file_name = "weekly_message_20220313.csv"
    text_augment = TextAugment(folder_path, file_name)

    #text =  "類似するデータを生成する記事を書いてます。"
    
    regular_document = text_augment.regular_text()
    with open(folder_path+'visual_text.txt', 'w', encoding='utf-8') as f:
        for d in regular_document:
            f.write(f"{d}\n")
    '''
    document_train, document_test = train_test_split(regular_document, test_size=0.2)
    document_dev = document_test[:len(document_test)]
    document_test = document_test[len(document_test):]
    with open(folder_path+'dev_set.txt', 'w', encoding='utf-8') as f:
        for d in document_dev:
            f.write(f"{d}\n")

    with open(folder_path+'test_set.txt', 'w', encoding='utf-8') as f:
        for d in document_dev:
            f.write(f"{d}\n")

    with open(folder_path+'train_set.txt', 'w', encoding='utf-8') as f:
        for text in tqdm(document_train):
            if text == "":
                continue
            augmented_sentences = text_augment.eda(text, alpha_sr=0.2, alpha_ri=0.1, alpha_rs=0.05, p_rd=0.05, num_aug=5)
            for d in augmented_sentences:
                f.write(f"{d}\n")

            augmented_sentences = text_augment.eda(text, alpha_sr=0.15, alpha_ri=0.15, alpha_rs=0.05, p_rd=0.05, num_aug=5)
            for d in augmented_sentences:
                f.write(f"{d}\n")
            
            augmented_sentences = text_augment.eda(text, alpha_sr=0.1, alpha_ri=0.15, alpha_rs=0.05, p_rd=0.05, num_aug=2)
            for d in augmented_sentences:
                f.write(f"{d}\n")
            
            augmented_sentences = text_augment.eda(text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.05, num_aug=2)
            for d in augmented_sentences:
                f.write(f"{d}\n")
    '''
    
    print("完了")