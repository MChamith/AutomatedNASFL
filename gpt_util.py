import re
import openai

def initialize_configs(no_of_clients):
    system_content = '''You are an AI that strictly conforms to responses in Python. You are an assistant in providing neural network architectures as 
search space for neural architecture search.'''


    user_input = '''I need help designing the optimal CNN architecture for a specific dataset. I plan to start with
    few variety of models and change the model architecture based on model performance. As the initial step,
    please provide {no_client} CNN architectures with varying width, depth, and layer types.The input data is a 3x32x32 image with 10 classification labels. There are altogether
    15000 datapoints for training. When suggesting models give importance to above factors to make the model complexity
    accordingly. After you suggest a design, I will test its actual performance and provide you with feedback. Based
    on the results of previous experiments, we can collaborate to iterate and improve the design. Please avoid
    suggesting the same design again during this iterative process.'''.format(no_client=no_of_clients)

    suffix = '''Your responses should contain valid Python code only, with no additional comments, explanations, 
    or dialogue. Provide the PyTorch models according to the given prompt. Important!! Output the solution as {no_client} 
    different Python code bases.Start each code base with <Code> and end code base with </Code> brackets. So there 
    must be {no_client} brackets. Include PyTorch imports with `import torch`, `from torch import nn`, and `import 
    torch.nn.functional as F`. Do not include model initialization code. '''.format(no_client = no_of_clients)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_input + suffix}, ]

    res = openai.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=messages, temperature=0, n=1).choices[0].message

    messages.append(res)
    gpt_output = res.content
    print(gpt_output)
    pattern = r"<Code>(.*?)</Code>"
    matches = re.findall(pattern, gpt_output, re.DOTALL)

    # Strip whitespace and print each code block
    code_strings = [match.strip() for match in matches]

    if len(code_strings) != no_of_clients:
        print('not enough models rerunning initialize')
        return initialize_configs(no_of_clients)

    return code_strings, messages


def intermediate_configs(messages, accuracy, model, no_of_clients):
    user_input = '''By using provided model {model}, we achieved an loss of {acc}%. As previous Please recommend {m_no} 
    new models that outperforms prior architectures based on the above mentioned experiments on neural 
    architecture search. Take step towards making the model differ from above by making model complex. Or any 
    other means like using different architecture, adding more layers, increasing parameters etc. Output should 
    be structured same as previous case. As in the previous case the input is torch.zeros((1,3,32,32)) and output a 
    tensor of size torch.size((1,10)). Please make sure that all the dimensions are correct in the network and no 
    error will arise'''.format(acc=accuracy, model=model, m_no =  no_of_clients)

    messages.append({"role": "user", "content": user_input})

    res = openai.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=messages, temperature=0, n=1).choices[0].message

    messages.append(res)
    gpt_output = res.content
    # print(gpt_output)
    pattern = r"<Code>(.*?)</Code>"
    matches = re.findall(pattern, gpt_output, re.DOTALL)

    # Strip whitespace and print each code block
    code_strings = [match.strip() for match in matches]
    del messages[-1]
    if len(code_strings) != no_of_clients:
        del messages[-1]
        print('not enough models rerunning intermediate')
        return intermediate_configs(messages, accuracy, model, no_of_clients)

    return code_strings, messages


def handle_error(messages, model, err_msg):
    user_input = '''The suggested model {model} gives the following error.\n {err}. \n Please suggest a new model 
    architecture with similar parameter complexity that conforms with the dimensions of the input and output 
    correctly to avoid above error.the input is torch.zeros((1,3,32,32)) and output a tensor of size torch.size((1,
    10)) '''
    user_input = user_input.format(err=str(err_msg), model=model)
    messages.append({"role": "user", "content": user_input})

    res = openai.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=messages, temperature=0.6, n=1).choices[0].message

    # messages.append(res)
    gpt_output = res.content

    pattern = r"<Code>(.*?)</Code>"
    matches = re.findall(pattern, gpt_output, re.DOTALL)

    # Strip whitespace and print each code block
    code_strings = [match.strip() for match in matches]
    del messages[-1]
    return code_strings, messages
