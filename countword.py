import re
from collections import defaultdict

def wordcount(text):
    # 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)
    
    # 转换为小写
    text = text.lower()
    
    # 分割单词
    words = text.split()
    
    # 统计单词出现的次数
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1
        
    return dict(word_counts)

# 测试示例
input_text = """Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!"""
input_text2 = """Got this panda plush toy for my daughter's birthday,
who loves it and takes it everywhere. It's soft and
super cute, and its face has a friendly look. It's
a bit small for what I paid though. I think there
might be other options that are bigger for the
same price. It arrived a day earlier than expected,
so I got to play with it myself before I gave it
to her.
"""
print(wordcount(input_text))
print(wordcount(input_text2))