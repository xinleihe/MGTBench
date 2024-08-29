## Task 2

### Concat with first half prompt


| Category               | LLM      | LM-D  | ll-GPT2 |
| ---------------------- | -------- | ----- | ------- |
| Physics                | Moonshot | 0.65  | 0.515   |
| Medicine               | Moonshot | 0.936 | 0.794   |
| Biology                | Moonshot | 0.655 | 0.627   |
| Electrical_engineering | Moonshot | 0.725 | 0.598   |
| Computer_science       | Moonshot | 0.66  | 0.589   |
| Literature             | Moonshot | 0.765 | 0.82    |
| History                | Moonshot | 0.857 | 0.902   |
| Education              | Moonshot | 0.649 | 0.824   |
| Art                    | Moonshot | 0.643 | Na      |
| Law                    | Moonshot | 0.877 | 0.802   |
| Management             | Moonshot | 0.939 | 0.826   |
| Philosophy             | Moonshot | 0.815 | 0.843   |
| Economy                | Moonshot | 0.665 | 0.671   |
| Math                   | Moonshot | 0.67  | 0.548   |
| Statistics             | Moonshot | 0.65  | 0.649   |
| Chemistry              | Moonshot | 0.848 | 0.71    |


**LM-D**: 

| Category               | LLM      | Seed   | cut_length | num_data | F1    |
| ---------------------- | -------- | ------ | ---------- | -------- | ----- |
| Physics                | Moonshot | 420    | 6000       | 3000     | 0.65  |
| Medicine               | Moonshot | 420    | 6000       | 3000     | 0.936 |
| Biology                | Moonshot | 420    | 3000       | 3000     | 0.655 |
| Electrical_engineering | Moonshot | 114514 | 6000       | 3000     | 0.725 |
| Computer_science       | Moonshot | 114514 | 6000       | 6000     | 0.66  |
| Literature             | Moonshot | 1000   | 6000       | 6000     | 0.765 |
| History                | Moonshot | 1000   | 3000       | 6000     | 0.857 |
| Education              | Moonshot | 1000   | 6000       | 3000     | 0.649 |
| Art                    | Moonshot | 114514 | 3000       | 6000     | 0.643 |
| Law                    | Moonshot | 3407   | 6000       | 3000     | 0.877 |
| Management             | Moonshot | 420    | 3000       | 3000     | 0.939 |
| Philosophy             | Moonshot | 3407   | 6000       | 3000     | 0.815 |
| Economy                | Moonshot | 420    | 6000       | 3000     | 0.665 |
| Math                   | Moonshot | 114514 | 3000       | 3000     | 0.67  |
| Statistics             | Moonshot | 3407   | 3000       | 3000     | 0.65  |
| Chemistry              | Moonshot | 3407   | 3000       | 6000     | 0.848 |


### Using Generated text only

| Category               | LLM      | LM-D  | ll-GPT2 |
| ---------------------- | -------- | ----- | ------- |
| Physics                | Moonshot | 0.65  | 0.744   |
| Medicine               | Moonshot | 0.953 | 0.915   |
| Biology                | Moonshot | 0.655 | 0.842   |
| Electrical_engineering | Moonshot | 0.663 | 0.828   |
| Computer_science       | Moonshot | 0.661 | 0.783   |
| Literature             | Moonshot | 0.733 | 0.931   |
| History                | Moonshot | 0.668 | 0.956   |
| Education              | Moonshot | 0.904 | 0.932   |
| Art                    | Moonshot | 0.645 | Na      |
| Law                    | Moonshot | 0.893 | 0.925   |
| Management             | Moonshot | 0.962 | 0.932   |
| Philosophy             | Moonshot | 0.844 | 0.93    |
| Economy                | Moonshot | 0.654 | 0.879   |
| Math                   | Moonshot | 0.647 | 0.662   |
| Statistics             | Moonshot | 0.65  | 0.712   |
| Chemistry              | Moonshot | 0.963 | 0.762   |


## Task 3

| Category               | LLM      | LM-D  | ll-gpt2 | ll-llama2 |
| ---------------------- | -------- | ----- | ------- | --------- |
| Physics                | Moonshot | 0.848 | 0.543   | 0.595     |
| Medicine               | Moonshot | 0.66  | 0.87    | 0.852     |
| Biology                | Moonshot | 0.889 | 0.69    | 0.674     |
| Electrical_engineering | Moonshot | 0.874 | 0.648   | 0.711     |
| Computer_science       | Moonshot | 0.785 | 0.6     | 0.676     |
| Literature             | Moonshot | 0.692 | 0.767   | 0.722     |
| History                | Moonshot | 0.671 | 0.697   | 0.655     |
| Education              | Moonshot | 0.856 | 0.87    | 0.867     |
| Art                    | Moonshot | 0.65  | Na      | 0.731     |
| Law                    | Moonshot | 0.805 | 0.819   | 0.824     |
| Management             | Moonshot | 0.881 | 0.874   | 0.88      |
| Philosophy             | Moonshot | 0.907 | 0.785   | 0.746     |
| Economy                | Moonshot | 0.896 | 0.72    | 0.723     |
| Math                   | Moonshot | 0.85  | 0.664   | 0.493     |
| Statistics             | Moonshot | 0.757 | 0.622   | 0.536     |
| Chemistry              | Moonshot | 0.931 | 0.802   | 0.801     |

**Details of LM-D**

| Category               | LLM      | Method | Seed   | cut_length | num_data | F1    |
| ---------------------- | -------- | ------ | ------ | ---------- | -------- | ----- |
| Physics                | Moonshot | LM-D   | 420    | 3000       | 3000     | 0.848 |
| Medicine               | Moonshot | LM-D   | 420    | 5000       | 3000     | 0.66  |
| Biology                | Moonshot | LM-D   | 114514 | 3000       | 5000     | 0.889 |
| Electrical_engineering | Moonshot | LM-D   | 3407   | 3000       | 5000     | 0.874 |
| Computer_science       | Moonshot | LM-D   | 114514 | 3000       | 5000     | 0.785 |
| Literature             | Moonshot | LM-D   | 420    | 5000       | 3000     | 0.692 |
| History                | Moonshot | LM-D   | 3407   | 3000       | 5000     | 0.671 |
| Education              | Moonshot | LM-D   | 3407   | 3000       | 3000     | 0.856 |
| Art                    | Moonshot | LM-D   | 420    | 3000       | 3000     | 0.65  |
| Law                    | Moonshot | LM-D   | 114514 | 3000       | 5000     | 0.805 |
| Management             | Moonshot | LM-D   | 420    | 5000       | 5000     | 0.881 |
| Philosophy             | Moonshot | LM-D   | 420    | 3000       | 3000     | 0.907 |
| Economy                | Moonshot | LM-D   | 114514 | 3000       | 5000     | 0.896 |
| Math                   | Moonshot | LM-D   | 3407   | 3000       | 3000     | 0.85  |
| Statistics             | Moonshot | LM-D   | 42     | 5000       | 5000     | 0.757 |
| Chemistry              | Moonshot | LM-D   | 3407   | 5000       | 3000     | 0.931 |


## Old data 

| Dataset   | Method          | ChatGLM   | Dolly   | ChatGPT-turbo   | GPT4All   | StableLM   | Claude   |
| --------- | --------------- | --------- | ------- | --------------- | --------- | ---------- | -------- |
| Essay     | ll(Llama2)      | 0.967     | 0.767   | 0.985           | 0.860     | 0.419      | 0.928    |
|           | ll(gpt-2)       | 0.970     | 0.866   | 0.968           | 0.923     | 0.665      | 0.834    |
|           | LM-D            | 0.997     | 0.997   | 0.911           | 1.0       | 0.991      | 0.916    |
| --------- | --------------- | --------- | ------- | --------------- | --------- | ---------- | -------  |
| WP        | ll(llama2)      | 0.975     | 0.721   | 0.947           | 0.899     | 0.688      | 0.927    |
|           | ll(gpt-2)       | 0.980     | 0.794   | 0.841           | 0.934     | 0.788      | 0.773    |
|           | LM-D            | 0.667     | 0.935   | 0.930           | 0.941     | 0.915      | 0.667    |
| --------- | --------------- | --------- | ------- | --------------- | --------- | ---------- | -------- |
| Reuters   | ll(llama2)      | 0.952     | 0.574   | 0.964           | 0.489     | 0.572      | 0.955    |
|           | ll(gpt-2)       | 0.972     | 0.385   | 0.931           | 0.699     | 0.657      | 0.798    |
|           | LM-D            | 0.993     | 0.993   | 0.667           | 0.998     | 0.990      | 0.667    |

