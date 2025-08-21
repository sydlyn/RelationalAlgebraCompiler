# RACompiler - *Relational Algebra Compiler*

### Overview

RACompiler is a tool to parse and execute relational algebra queries by translating them into SQL and running against a MySQL database.


### Requirements

- Python 3.8 or higher
- MySQL database accessible with proper credentials



## Setup Instructions

### 1. Clone the repository

```shell
git clone https://github.com/sydlyn/RelationalAlgebraCompiler.git
cd RACompiler
```


### 2. Create and activate a virtual environment

On macOS/Linux: 
```shell
python -m venv venv
source ./venv/bin/activate
```
On Windows: 
```shell
venv\Scripts\activate
```

### 3. Install dependencies and the package

```shell
pip install flit
flit install --symlink
```

This will install RACompiler in editable mode and register the `rac` command.

### 4. Configure your database connection

Create a `.env` file in the project root with your MySQL credentials:
```
DB_HOST=your_host  
DB_PORT=your_port           # defaults to 3306
DB_USER=your_user  
DB_PASSWORD=your_password  
DB_NAME=your_database  
```
---

### Usage

Run the RACompiler command line interface:

```shell
rac
```

## Running Tests

Make sure your database is running and accessible.

Run all tests with:
```shell
python -m unittest discover -s tests -p "test*.py"
```

Or if you use pytest:
```shell
pytest -q 
```

---

### Project Structure
```
RACompiler/  
├── ra_compiler/        # Main package source code  
│   ├── __init__.py  
│   ├── cli.py          # Command line interface entry point  
│   ├── mysql.py        # MySQL connection code  
│   ├── parser.py       # Query parsing logic  
│   ├── translator.py   # Query translation to /IR  
│   ├── executor.py     # Query execution  
│   ├── utils.py        # Helper functions  
│   └── grammar.lark    # Query grammar definition  
├── tests/              # Test suite  
│   └── ...             
├── .env                # MySQL credentials (gitignored)  
├── pyproject.toml      # Build configuration using flit  
├── README              # This file  
└── ...
```

---

### Final Notes

- Make sure your MySQL server is up and the `.env` file is correctly set before running.
- `flit install --symlink` ensures that code changes are immediately available without reinstalling.

