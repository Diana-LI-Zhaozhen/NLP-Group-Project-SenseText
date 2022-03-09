---
Title: From Messy to Clean: Our Way to Deal with 10-K
Date: 2022-03-09 15:11
Category: Progress Report
---

By Group SenseText

We have currently finished text processing and obtained our BoW. Our texts are from the items in 10-K. The volume of the words is huge, and the processing is time-consuming. While struggling with bugs, we learned a lot from them and would like to share our experience with the whole class. 

# The List Problem
This problem is caused by the confusions in object reference. A list stores a set of references (or paths) to the objects in computer memory. For example, list A in the picture below points to the numbers 1, 2, 3 in memory. If we simply copy list A to list B using the operator `=`, list B will only copy the object references, i.e., B points to the same numbers in memory as A.

![object reference]({static}/images/list_problem.jpeg)

Hence, if we modify number 1 following the reference in A, we will also change the first element in B. To copy A without the reference, we can use the `.copy` method.

In the context of text cleaning, if we want to remove some words from a list, it is better to use a copy of the list as the iterable. Otherwise, the iteration and deletion would happen at the same time and create some trouble. The codes below demonstrate the issue.

```python
# Create a list A
A = ['Alliteration','Example','Peter','Piper','picked','a','peck','of','pickled','peppers']

# Remove all words starting with P/p
for i in A:
    if i.lower().startswith('p'):
        A.remove(i)

print(A)  # ['Alliteration', 'Example', 'Piper', 'a', 'of', 'peppers']

# 'Piper' & 'peppers' are still there because iteration and deletion happen at the same time.
# Once 'Peter' is deleted, 'Piper' takes the 3rd position. 
# However, the iterator has already checked the 3rd position. So it skips 'Piper.'
# To see this, let's print each i
A = ['Alliteration','Example','Peter','Piper','picked','a','peck','of','pickled','peppers']

# Remove all words starting with P/p
for i in A:
    print(i)
    if i.lower().startswith('p'):
        A.remove(i)

# Alliteration
# Example
# Peter
# picked
# peck
# pickled
## No 'Piper' & 'peppers'

# To solve this, we iterate through a copy of A
A = ['Alliteration','Example','Peter','Piper','picked','a','peck','of','pickled','peppers']

for i in A.copy():      # copy A
    if i.lower().startswith('p'):
        A.remove(i)

print(A)      # ['Alliteration', 'Example', 'a', 'of']
```


# The XX Problem


# The XX Problem


website link: 
This is a website with link [Pelican Documentation](https://docs.getpelican.com/en/latest/).

quote:
>Fed watching is a great tool to make money. I have been making all my
>gazillions using this technique.
> LoL






