# 🚀 fast-dedupe

[![PyPI version](https://img.shields.io/pypi/v/fast-dedupe.svg)](https://pypi.org/project/fast-dedupe/)
[![Python Versions](https://img.shields.io/pypi/pyversions/fast-dedupe.svg)](https://pypi.org/project/fast-dedupe/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/username/fast-dedupe/tests.yml?branch=main)](https://github.com/username/fast-dedupe/actions)
[![codecov](https://codecov.io/gh/username/fast-dedupe/branch/main/graph/badge.svg)](https://codecov.io/gh/username/fast-dedupe)
[![License](https://img.shields.io/github/license/username/fast-dedupe.svg)](https://github.com/username/fast-dedupe/blob/main/LICENSE)

**Fast, Minimalist Text Deduplication Library for Python**

## 🧩 Problem Statement

Developers frequently face **duplicate textual data** when dealing with:

- User-generated inputs (reviews, comments, feedback)
- Product catalogs (e-commerce)
- Web-scraping (news articles, blogs, products)
- CRM data (customer profiles, leads)
- NLP/AI training datasets (duplicate records skew training)

**Existing Solutions and their Shortcomings:**

- **Manual Deduplication:** Slow, error-prone, impractical at scale.
- **Pandas built-in methods:** Only exact matches; ineffective for slight differences (typos, synonyms).
- **Fuzzywuzzy / RapidFuzz:** Powerful but require boilerplate setup for large-scale deduplication.

**Solution:**  
A simple, intuitive, ready-to-use deduplication wrapper around RapidFuzz, minimizing setup effort while providing great speed and accuracy out-of-the-box.

## ⚡ Installation

```bash
pip install fast-dedupe
```

## 🚀 Quick Start

```python
import fastdedupe

data = ["Apple iPhone 12", "Apple iPhone12", "Samsung Galaxy", "Samsng Galaxy", "MacBook Pro", "Macbook-Pro"]

# One-liner deduplication
clean_data, duplicates = fastdedupe.dedupe(data, threshold=85)

print(clean_data)
# Output: ['Apple iPhone 12', 'Samsung Galaxy', 'MacBook Pro']

print(duplicates)
# Output: {'Apple iPhone 12': ['Apple iPhone12'], 
#          'Samsung Galaxy': ['Samsng Galaxy'], 
#          'MacBook Pro': ['Macbook-Pro']}
```

## 📌 Key Features

- **High performance:** Powered by RapidFuzz for sub-millisecond matching
- **Simple API:** Single method call (`fastdedupe.dedupe()`)
- **Flexible Matching:** Handles minor spelling differences, hyphens, abbreviations
- **Configurable Sensitivity:** Adjust matching threshold easily
- **Detailed Output:** Cleaned records and clear mapping of detected duplicates

## 🎯 Use Cases

### E-commerce Catalog Management

```python
products = [
    "Apple iPhone 15 Pro Max (128GB)",
    "Apple iPhone-12",
    "apple iPhone12",
    "Samsung Galaxy S24",
    "Samsung Galaxy-S24",
]

cleaned_products, duplicates = fastdedupe.dedupe(products, threshold=90)

# cleaned_products:
# ['Apple iPhone 15 Pro Max (128GB)', 'Apple iPhone-12', 'Samsung Galaxy S24']

# duplicates identified clearly:
# {
#   'Apple iPhone-12': ['apple iPhone12'],
#   'Samsung Galaxy S24': ['Samsung Galaxy-S24']
# }
```

### Customer Data Management

```python
emails = ["john.doe@gmail.com", "john_doe@gmail.com", "jane.doe@gmail.com"]
clean, dupes = fastdedupe.dedupe(emails, threshold=95)

# clean → ["john.doe@gmail.com", "jane.doe@gmail.com"]
# dupes → {"john.doe@gmail.com": ["john_doe@gmail.com"]}
```

## 📖 API Reference

### `fastdedupe.dedupe(data, threshold=85, keep_first=True)`

Deduplicates a list of strings using fuzzy matching.

**Parameters:**
- `data` (list): List of strings to deduplicate
- `threshold` (int, optional): Similarity threshold (0-100). Default is 85.
- `keep_first` (bool, optional): If True, keeps the first occurrence of a duplicate. If False, keeps the longest string. Default is True.

**Returns:**
- `tuple`: (clean_data, duplicates)
  - `clean_data` (list): List of deduplicated strings
  - `duplicates` (dict): Dictionary mapping each kept string to its duplicates

## 👥 Target Audience

- **Data Engineers / Analysts:** Cleaning large datasets before ETL, BI tasks, and dashboards
- **ML Engineers & Data Scientists:** Cleaning datasets before training models to avoid bias and data leakage
- **Software Developers (CRM & ERP systems):** Implementing deduplication logic without overhead
- **Analysts (E-commerce, Marketing):** Cleaning and deduplicating product catalogs, customer databases

## 🛠️ Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
