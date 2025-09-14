
few_user_1 = "How long did Grand Blanc High School participate for?"
few_model_1 = """
Question: In which year did Guillermo García-López win more than 10 games?

Thought: From the summary, the **W–L** column indicates wins and losses. I must extract the wins by splitting the string before "-". Then I will check where wins > 10. Rows with "nan" must be ignored.

Action: python_repl_ast
Action Input:
df[df['W–L'].str.contains('-', na=False)]

Observation:
|    | Year | Tournament       | Surface | W–L  |
|---:|-----:|:-----------------|:--------|:-----|
|  3 | 2009 | Some Tournament  | Clay    | 12-9 |
|  7 | 2010 | Another Open     | Hard    | 14-11|
|  9 | 2011 | Example Masters  | Clay    | 8-10 |

Thought: Now I will filter to rows where wins > 10.

Action: python_repl_ast
Action Input:
df[df['W–L'].str.split('-', expand=True)[0].astype(int) > 10][['Year','W–L']]

Observation:
|    | Year | W–L  |
|---:|-----:|:-----|
|  3 | 2009 | 12-9 |
|  7 | 2010 | 14-11|

Thought: The years with wins > 10 are 2009 and 2010. The question asks "the year", so both should be listed.

Final Answer: 2009, 2010

"""

few_user_2 = "what was the total number of awards this film won?"
few_model_2 = """
Question: What was the total number of awards this film won?

Thought: From the summary, the **Awards** column shows either award names or 'nan'. The goal is to count the number of award-winning events (rows), not split into multiple lines. Rows with 'nan' must be ignored.

Action: python_repl_ast
Action Input:
df[df['Awards'].notna()][['Festival','Awards']]

Observation:
|    | Festival                                   | Awards                                                                                                             |
|---:|:-------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
|  0 | Santa Barbara International Film Festival  | Top 11 "Best of the Fest" Selection                                                                                |
|  2 | Fantasia Festival                          | Special Mention\\n"for the resourcefulness and unwavering determination by a director to realize his unique vision" |
|  3 | Athens International Film Festival         | Best Director                                                                                                      |
| 10 | Toronto After Dark                         | Best Special Effects\\nBest Musical Score                                                                           |

Thought: There are 4 non-null rows in the Awards column. Even though some rows list multiple prizes, they count as a single event. Therefore the total number of awards is 4.

Final Answer: 4
"""