"""
Example: Coverage Auditing

Compare your test suite against production traffic to discover semantic regions
that your tests have never covered. These are gaps — real user behavior your
evaluation suite is blind to.

How it works:
  1. Both test and production prompts are embedded
  2. For each production prompt, the nearest test prompt is found (k-NN)
  3. Production prompts farther than the threshold from any test prompt are "uncovered"
  4. Uncovered prompts are clustered into gap clusters
  5. Each gap cluster is named by an LLM

A low overall_coverage_score means your test suite is missing large portions
of what real users are asking.

Requirements:
    pip install faultmap[local,rich]
    export OPENAI_API_KEY=...
"""

from faultmap import SliceAnalyzer

# ---------------------------------------------------------------------------
# Your test suite — what you've explicitly evaluated
# ---------------------------------------------------------------------------

test_prompts = [
    # Password / account management
    "How do I reset my password?",
    "How do I change my email address?",
    "How do I update my billing information?",
    "How do I cancel my subscription?",
    "How do I delete my account?",

    # Basic product questions
    "What features does the Pro plan include?",
    "How much does the Enterprise plan cost?",
    "Can I use this on mobile?",
    "Is there a free trial available?",
    "What is your refund policy?",

    # Integration basics
    "Does this integrate with Slack?",
    "Can I connect to Salesforce?",
    "Is there an API available?",
    "Do you support SSO?",
    "What data formats can I import?",
] * 4  # 60 test prompts total

# ---------------------------------------------------------------------------
# Production traffic — what real users are actually asking
# ---------------------------------------------------------------------------

production_prompts = [
    # Covered topics (similar to test set)
    "I forgot my password, how do I get back in?",
    "Where do I update my credit card?",
    "How do I sign out of all devices?",
    "Can I change my username?",
    "How much does the basic tier cost per month?",
    "Do you offer student discounts?",
    "Can I export my data?",
    "How do I connect to my CRM?",
    "Is there a REST API?",
    "Can I use SAML for authentication?",

    # NOT in test set — authentication security (gap)
    "How do I set up two-factor authentication?",
    "How do I enable an authenticator app for my account?",
    "What happens if I lose access to my 2FA device?",
    "Can I use hardware security keys like YubiKey?",
    "How do I enforce 2FA for my entire organization?",
    "What 2FA methods do you support?",
    "Can I set up SMS-based two-factor authentication?",
    "How do I disable 2FA on my account?",
    "Is biometric authentication supported?",
    "How do I audit which users have 2FA enabled?",

    # NOT in test set — SSO / enterprise identity (gap)
    "How do I configure SAML SSO for my organization?",
    "Which identity providers do you support for SSO?",
    "How do I set up Okta integration?",
    "Can I use Azure Active Directory for login?",
    "How do I map SAML attributes to user roles?",
    "What happens to user accounts when SSO is disabled?",
    "How do I test my SSO configuration before going live?",
    "Can I have both SSO and password login enabled?",
    "How do I provision users via SCIM?",
    "What audit logs are available for SSO login events?",

    # NOT in test set — compliance and data privacy (gap)
    "Are you SOC 2 Type II certified?",
    "Where is my data stored geographically?",
    "Do you process data in the EU for GDPR compliance?",
    "Can I get a Data Processing Agreement (DPA)?",
    "What is your data retention policy?",
    "How do I submit a GDPR data deletion request?",
    "Do you support data residency requirements?",
    "Is your platform HIPAA compliant?",
    "What encryption standards do you use for data at rest?",
    "How do I export all my data for compliance purposes?",
] * 2  # 100 production prompts total

# ---------------------------------------------------------------------------
# Run coverage audit
# ---------------------------------------------------------------------------

analyzer = SliceAnalyzer(
    model="gpt-4o-mini",
    embedding_model="all-MiniLM-L6-v2",
    min_slice_size=5,
)

print("Auditing test suite coverage...")
print(f"Test prompts: {len(test_prompts)}")
print(f"Production prompts: {len(production_prompts)}\n")

coverage = analyzer.audit_coverage(
    test_prompts=test_prompts,
    production_prompts=production_prompts,
    # distance_threshold=None  ← auto-computed from the distance distribution
    min_gap_size=5,
)

print(coverage)

# ---------------------------------------------------------------------------
# Inspect gaps in detail
# ---------------------------------------------------------------------------

print(f"\n--- Coverage Summary ---")
print(f"Overall coverage score: {coverage.overall_coverage_score:.1%}")
print(f"Test prompts: {coverage.num_test_prompts}")
print(f"Production prompts: {coverage.num_production_prompts}")
print(f"Coverage gaps found: {coverage.num_gaps}")
print(f"Distance threshold used: {coverage.distance_threshold:.4f}")

if coverage.gaps:
    print(f"\n--- Gaps (sorted by severity) ---")
    for i, gap in enumerate(coverage.gaps, 1):
        print(f"\nGap #{i}: {gap.name!r}")
        print(f"  Description: {gap.description}")
        print(f"  Size: {gap.size} production prompts with no nearby test case")
        print(f"  Mean distance to nearest test prompt: {gap.mean_distance:.4f}")
        print(f"  Representative prompts:")
        for p in gap.representative_prompts[:3]:
            print(f"    - {p}")

    print(f"\n--- Action items ---")
    print("Add test prompts covering these topics:")
    for gap in coverage.gaps:
        print(f"  * {gap.name}: ~{gap.size} production prompts uncovered")
else:
    print("\nTest suite covers production traffic well. No significant gaps found.")
