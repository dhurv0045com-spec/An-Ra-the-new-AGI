# step20000 — global_step 20,000 — cuda

## A_conversation/chat_greeting
PROMPT: 'Hello! How are you today?'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## A_conversation/chat_identity
PROMPT: 'Who are you? Tell me about yourself.'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## A_conversation/chat_smalltalk
PROMPT: 'What do you like to do on a rainy day?'
OUTPUT: '\n-------------------------------'

## A_conversation/chat_help
PROMPT: 'Can you help me understand what photosynthesis is?'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## A_conversation/chat_persona
PROMPT: 'H: What is your name?\nANRA:'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## B_factual/fact_capital
PROMPT: 'The capital of France is'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'Paris' MATCH: False

## B_factual/fact_capital_ctx
PROMPT: '<k>The capital of Japan is Tokyo.</k>\n<q>What is the capital of Japan?</q>\n<answer>'
OUTPUT: '</github. the the the the the the the the the the the the the the'
GOLD: 'Tokyo' MATCH: False

## B_factual/fact_number
PROMPT: 'There are how many days in a week? Answer:'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'seven' MATCH: False

## B_factual/fact_color
PROMPT: 'The sky during a clear day is'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'blue' MATCH: False

## C_instruction/echo_ember
PROMPT: 'Echo exactly this word: ember'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'ember' MATCH: False

## C_instruction/echo_quartz
PROMPT: 'Echo exactly this word: quartz'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'quartz' MATCH: False

## C_instruction/copy_ctx
PROMPT: '<k>the magic word is lantern</k>\n<q>What is the magic word?</q>\n<answer>'
OUTPUT: '\n-------------------------------'
GOLD: 'lantern' MATCH: False

## C_instruction/repeat_number
PROMPT: 'Say exactly this number: forty-two'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'forty' MATCH: False

## D_arithmetic/arith_add
PROMPT: 'Compute 7 + 5.'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: '12' MATCH: False

## D_arithmetic/arith_mul
PROMPT: 'Compute (3 + 4) x 2.'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: '14' MATCH: False

## D_arithmetic/arith_toolresult
PROMPT: 'Use the calculator to add 20 and 22.\n<tool_output>42</tool_output>\nWhat is 20 + 22?'
OUTPUT: '\n-------------------------------'
GOLD: '42' MATCH: False

## E_toolcall/tool_request_calc
PROMPT: 'What is 458 times 12? If you need a calculator, reply with CALL calculator(458*12)'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'calculator' MATCH: False

## E_toolcall/tool_request_search
PROMPT: 'What happened in the news today? If you need search, reply with CALL search(news)'
OUTPUT: ' the the the the the the the the the the the the the the the the'
GOLD: 'search' MATCH: False

## F_code/code_def
PROMPT: 'def add_numbers(a, b):\n    return'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## F_code/code_for
PROMPT: 'for i in range(10):\n    print'
OUTPUT: ' the the the the the the the the the the the the the the the the'

## Sampling health
- ' but and $\\pm and and is happening by place on person' (rep=0.00)
- ' main them of the MS AT new “L-----' (rep=0.00)
- ' a s in quality the marked or with is and series health' (rep=0.00)
- ' $ However are of by the already you. example in maximum ' (rep=0.00)