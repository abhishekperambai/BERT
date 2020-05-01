"""
Make sure bert.py exists in the same directory as this script and the model files downloaded form the dropbox link is placed under the model folder
"""

from bert import QA


model = QA('model')

"""
Now, let us implement comprehending a passage from google with our existing zero shot model
"""

passage = 'There was a princess called Maggie, she was tall and as white as snow. She had red lips like a \
rose and her hair was brown. She had light blue eyes and was very nice and kind. \
She was in love with a bricklayer called Kevin. He was tall, had brown hair and tanned skin. \
He was strong, had dark eyes, was kind and he was never angry'

ques = 'What is the name of the princess?'


"""
Generally the predict method gives out a dictionary with other values as well, but since our interest lies with answer we subset the same.
"""
answer = model.predict(passage,ques)['answer']

print("{} : {}".format(ques,answer))

"""
Let us try few more questions to see the edge case scenarios to understand where the model might break
"""

answer = model.predict(passage, 'Who was Maggie?')['answer']

print("{} : {}".format('Who was Maggie?',answer))

answer = model.predict(passage, 'What is the color of the Maggies hair?')['answer']

print("{} : {}".format('What is the color of the Maggies hair?',answer))


"""
When the model fails to find an answer to the question posed it falls back on the default answer. 
"""


answer = model.predict(passage, 'Kevin')['answer']
answer_ = model.predict(passage, 'Maggie')['answer']

"""
In this the model yeilds same answer for both the questions below
"""
print("{} : {}".format('Kevin',answer))
print("{} : {}".format('Maggie',answer))

"""
In Conclusion: Since this is a model and not a human brain, there might be edge case scenarios as these where the model might break.
"""


 