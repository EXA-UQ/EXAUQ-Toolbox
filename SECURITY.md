# Security Policy

## Supported Versions

Only the latest release of the EXAUQ-Toolbox receives security updates.

## Reporting a Vulnerability

Please do not open a public issue for security vulnerabilities. Instead, use
GitHub's private vulnerability reporting: go to the repository's
[Security tab](https://github.com/EXA-UQ/EXAUQ-Toolbox/security) and choose
"Report a vulnerability". Include the affected version and steps to reproduce.
You can expect an initial response within two weeks.

## Automated Security Measures

- Dependabot alerts and grouped weekly version updates, with a 7-day cooldown
  on new releases to guard against compromised package versions
- pip-audit dependency scans on every pull request and on a weekly schedule
- GitHub dependency review on pull requests
- Secret scanning via a gitleaks pre-commit hook and GitHub push protection
- All GitHub Actions pinned by commit SHA
