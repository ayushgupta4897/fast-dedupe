# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-04-01

### Added
- Multiple string similarity algorithms:
  - Levenshtein (original algorithm)
  - Jaro-Winkler for name matching
  - Cosine similarity with character n-grams for document comparison
  - Jaccard similarity for set-based text analysis
  - Soundex/phonetic matching for handling pronunciation variations
- New `similarity_algorithm` parameter for the `dedupe` function
- Direct access to individual similarity functions
- Visualization tools:
  - `visualize_similarity_matrix` for comparing multiple strings
  - `visualize_algorithm_comparison` for comparing algorithm performance
- New CLI options:
  - `-a/--algorithm` to select similarity algorithm
  - `--visualize` to generate visualizations
  - `--compare-algorithms` to compare all algorithms
- New example script `similarity_comparison.py`
- Comprehensive documentation for all new features
- Test suite for the new similarity algorithms

### Changed
- Updated core deduplication logic to support multiple similarity algorithms
- Improved parallel processing to work with all similarity algorithms
- Enhanced documentation with algorithm selection guidelines
- Updated requirements to include additional dependencies for new features

### Requirements
- Python 3.8 or higher
- RapidFuzz 2.0.0 or higher
- Jellyfish 1.0.0 or higher (for Soundex)
- Matplotlib 3.5.0 or higher (for visualization)
- Scikit-learn 1.0.0 or higher (for cosine similarity)

## [0.1.0] - 2024-03-11

### Added
- Initial release of fast-dedupe
- Core deduplication functionality with configurable threshold and keep_first options
- Command-line interface for deduplicating files
- Support for different file formats (TXT, CSV, JSON)
- Comprehensive test suite with 93%+ code coverage
- Documentation and examples

### Requirements
- Python 3.8 or higher
- RapidFuzz 2.0.0 or higher 