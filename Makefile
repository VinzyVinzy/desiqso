# Variables
POETRY = poetry
PYTHON = python
PYTHON_VERSION = 3.12
DESIQSO = src/desiqso
DEBUG = src/debug

# =================
# Project commands
# =================

# Reset project command
reset:
	$(POETRY) env remove --all
	$(POETRY) env use $(PYTHON_VERSION)
	$(POETRY) install

# Install project command
install:
	$(POETRY) install

# ================
# Utility commands
# ================

# Run the test code
test:
	$(PYTHON) $(DESIQSO)/utils/test.py


# ================
# Runner commands
# ================

# Debug analysis command
run-debug:
	$(PYTHON) $(DEBUG)/run_debug.py

# Spectra loader command
download:
	$(PYTHON) $(DESIQSO)/pipelines/run_download.py

# Generate synthetic profiles command
generate-synthetic-profiles:
	$(PYTHON) $(DESIQSO)/pipelines/run_synthetic_profiles.py

# Spectra plotter command
plot-spectra:
	$(PYTHON) $(DESIQSO)/pipelines/run_spectra_plotter.py

# Stacked spectra plotter command
spectra-stacker:
	$(PYTHON) $(DESIQSO)/pipelines/run_spectra_stacker.py

# Statistics analysis command
plot-statistics:
	$(PYTHON) $(DESIQSO)/pipelines/run_statistics_plotter.py

# Program evolution command
dependencies-analysis:
	$(PYTHON) $(DESIQSO)/pipelines/run_dependency.py

# Sample statistics command
sample-statistics:
	$(PYTHON) $(DESIQSO)/pipelines/run_sample_statistics.py

# Run mock analysis command
run-mock-analysis:
	$(PYTHON) $(DESIQSO)/pipelines/run_mock_analysis.py

# Run completeness analysis command
completeness-analysis:
	$(PYTHON) $(DESIQSO)/pipelines/run_completeness.py

# Run physical analysis command
physical-analysis:
	$(PYTHON) $(DESIQSO)/pipelines/run_physical_analysis.py

# Export analysis results command
export-results:
	$(PYTHON) $(DESIQSO)/pipelines/run_export.py

# Run analysis command
run-analysis:
	$(PYTHON) $(DESIQSO)/pipelines/run_analysis.py