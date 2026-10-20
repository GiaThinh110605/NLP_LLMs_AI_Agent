# 00. Đảo ngược xâu ký tự
# Hãy đảo ngược xâu ký tự "stressed" (theo thứ tự từ cuối xâu đến đầu xâu ký tự).
text1 = "stressed"
print("Bài 1:", text1[::-1])

# 01. Trích xuất ký tự từ xâu ký tự
# Từ xâu ký tự "MPyaktQrBoilk RCSahr", 
# hãy trích xuất các ký tự ở vị trí 2,4,6,8,10,12,14,16,18,20 và 
# kết hợp theo thứ tự đó để tạo thành 1 xâu ký tự mới (ký tự space cũng được tính, các ký tự được đánh số từ 1).
text2 = "MPyaktQrBoilk RCSahr"
n_2 = len(text2)
sau_ky_tu = []
for i in range(2, n_2, 2):
    sau_ky_tu.append(text2[i])
print("Bài 2:", "".join(sau_ky_tu))

# 02. Kết hợp hai xâu ký tự
# Hãy kết hợp hai xâu ký tự "Partrol" và "Car" để tạo thành xâu mới "PatrolCar".
text3 = "Partrol"
text4 = "Car"
print("Bài 3: ", text3 + text4)

# 03. Tokenize và thống kê số lượng ký tự của mỗi từ
# Tokenize câu sau: "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
# Đưa ra danh sách gồm số ký tự alphabet trong mỗi từ theo thứ tự xuất hiện của từ đó trong câu.
import re
text5 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

text_after_split = re.findall("[a-zA-Z]+", text5)
arr1 = []
for txt in text_after_split:
    arr1.append(len(txt))
print("Bài 4:", arr1)

# 04. Ký tự thành phần
# Tokenize câu sau: "Hi He Lied Because Boron Could Not Oxidize Fluorine. 
# New Nations Might Also Sign Peace Security Clause. Arthur King Can."

# Lấy ra ký tự đầu tiên của các từ ở vị trí 1, 5, 6, 7, 8, 9, 15, 16, 19; 
# với các từ còn lại lấy ra 2 ký tự đầu tiên. 
# Tạo ra một map từ các xâu ký tự được trích ra tới vị trí của từ trong câu.

vi_tri = {1, 5, 6, 7, 8, 9, 15, 16, 19}
text_6 = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
text_after_split = re.findall("[a-zA-Z]+", text_6)
arr_2 = {}

for i, txt in enumerate(text_after_split, start=1):
    if i in vi_tri:
        arr_2[txt[0]] = i
    else:
        arr_2[txt[:2]] = i
print("Bài 5:", arr_2)

# 05. n-gram
# Viết hàm sinh ra tất cả các n-gram từ một dãy cho trước (xâu ký tự hoặc danh sách).

# Sử dụng hàm đã viết, sinh ra word bi-gram và character bi-gram từ câu sau: "I am an NLPer"

def n_gram(arr, n):
    word_bi_gram = [arr[i:i + n] for i in range(len(arr)- n + 1)]
    return word_bi_gram

text6 = "I am an NLPer"
print("Bài 6:", n_gram(text6.split(), 2))