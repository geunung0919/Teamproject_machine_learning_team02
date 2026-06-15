# Feature Engineering and Dataset Construction

This component is responsible for building the master dataset and defining stage-specific feature column policies. 

## Design Principles

- **Separation of Concerns**: Feature engineering (lag calculations, rolling averages, haversine isolation distances, and grade flow checks) is executed at the dataset/feature construction phase, prior to training modeling experiments.
- **Eligibility Policies**: School eligibility criteria (such as standard model eligibility, coordinate validation flags, and grade flow match flags) are explicitly computed here, ensuring model views are clean and free of anomalous/extreme events.
- **Stage Policies**: Exposes stage column lists dynamically ensuring that data-leakage keywords are filtered.


