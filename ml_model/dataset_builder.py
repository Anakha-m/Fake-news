"""
Dataset Builder & Curator for Fake News Detection System.
Generates an extensive, verified, multi-domain dataset covering real news
(Indian & International news, science, governance, economy, health, space, welfare policies)
and documented fact-checked misinformation (WhatsApp hoaxes, cash schemes, free recharge, miracle cures, conspiracies).
"""

import os
import pandas as pd

REAL_LABEL = 0
FAKE_LABEL = 1

REAL_CORPUS = [
    # Indian News, Governance, Policy & Economy
    ("Government Launches Pradhan Mantri Awas Yojana Urban 2.0 with Direct Subsidy Credit",
     "The Union Cabinet approved the implementation of PMAY-Urban 2.0 to provide financial assistance and interest subsidies to middle-class and economically weaker families for constructing affordable urban housing through direct bank transfers.", "Governance"),
    ("Reserve Bank of India Issues Official Guidelines on Digital Banking Security and Phishing Protection",
     "The Reserve Bank of India directed all commercial scheduled banks to implement multi-factor authentication and real-time fraud monitoring for online transactions, advising consumers never to share OTPs or click unverified links.", "Banking & Security"),
    ("ISRO Successfully Launches Navigation Satellite with Advanced Atomic Clocks from Sriharikota",
     "The Indian Space Research Organisation (ISRO) successfully launched its next-generation navigation satellite aboard GSLV rocket from Sriharikota spaceport, enhancing regional navigation accuracy across the subcontinent.", "Space & Science"),
    ("Reserve Bank of India Keeps Repo Rate Unchanged at 6.5 Percent Amid Steady Economic Growth",
     "The Monetary Policy Committee of the Reserve Bank of India voted to keep the benchmark repo rate steady at 6.5 percent, citing stable macroeconomic fundamentals and consistent domestic demand.", "Economy"),
    ("Kerala Government Launches Smart Water Grid Monitoring Project in Kochi Municipal Corporation",
     "The Kerala State Water Authority commissioned a digital sensor-based pipeline monitoring network in Kochi to detect leakage and ensure equitable municipal water distribution across coastal wards.", "Governance"),
    ("India Meteorological Department Issues Yellow Alert for Heavy Rains in Coastal Districts",
     "The India Meteorological Department issued a yellow weather advisory for several southern districts, predicting heavy monsoon spells and advising fishermen against venturing into rough seas.", "Environment"),
    ("Supreme Court of India Mandates Strict Environmental Impact Assessments for Infrastructure",
     "A three-judge bench of the Supreme Court of India mandated stringent ecological oversight and timely public hearings before granting clearances for major infrastructure projects in eco-sensitive zones.", "Governance"),
    ("Kerala University Inks Memorandum of Understanding for AI and Renewable Energy Research",
     "The University of Kerala signed an academic collaboration agreement with premier research institutes to develop decentralized solar microgrid controllers using machine learning.", "Education & Tech"),
    ("Ministry of Railways Completes 100 Percent Overhead Electrification of Key Broad Gauge Routes",
     "Indian Railways announced the completion of overhead track electrification across major railway zones, significantly reducing diesel traction reliance and lowering carbon emissions.", "Infrastructure"),
    ("National Health Authority Expands Digital Health Records Integration Across District Hospitals",
     "Under the Ayushman Bharat Digital Mission, over 50,000 public healthcare facilities have successfully linked electronic health records to facilitate secure patient medical histories.", "Health"),
    ("Finance Ministry Clarifies Income Tax Slabs and Thresholds for New Simplified Tax Regime",
     "The Central Board of Direct Taxes issued a comprehensive circular detailing tax rebate limits under Section 87A for salaried taxpayers earning up to 7 lakh rupees under the revised fiscal framework.", "Economy"),
    ("University Grants Commission Announces Revised Fellowship Guidelines for Higher Education",
     "The University Grants Commission notified updated junior research fellowship stipends and direct benefit disbursement procedures for doctoral candidates in central universities.", "Education"),
    ("National Payments Corporation of India Reports Record High UPI Monthly Transactions",
     "The National Payments Corporation of India confirmed that unified payments interface transactions reached a new milestone of 14 billion monthly payments processed across partner banks.", "Economy & Tech"),
    ("Ministry of Road Transport and Highways Inaugurates High-Speed Expressway Corridor",
     "The Union Minister for Road Transport and Highways opened a 240-kilometer access-controlled greenfield expressway stretch designed to reduce inter-city logistics transit times.", "Infrastructure"),
    ("SEBI Strengthens Disclosure Norms for Foreign Portfolio Investors and Beneficial Ownership",
     "The Securities and Exchange Board of India mandated stricter beneficial ownership disclosure guidelines for high-risk foreign portfolio investment funds operating in domestic equity markets.", "Economy & Finance"),
    ("Cabinet Committee on Economic Affairs Approves Minimum Support Prices for Kharif Crops",
     "The Cabinet Committee on Economic Affairs approved increased minimum support prices for major kharif agricultural crops to ensure remunerative returns for domestic farming communities.", "Agriculture & Policy"),
    ("Election Commission of India Publishes Comprehensive Electoral Roll Revision Schedule",
     "The Election Commission of India notified the timeline for the annual special summary revision of electoral rolls across assembly constituencies, enabling newly eligible voters to register.", "Governance"),
    ("Bureau of Indian Standards Releases Mandatory Quality Standards for Solar Photovoltaic Modules",
     "The Bureau of Indian Standards enacted mandatory certification protocols for domestic solar power components to ensure electrical safety and operational efficiency under tropical climates.", "Energy & Standards"),

    # Space, Physics & Global Science
    ("NASA James Webb Space Telescope Observes Most Distant Known Galaxy in Cosmic Dawn", 
     "Astronomers using NASA's James Webb Space Telescope have confirmed spectroscopic observations of galaxy JADES-GS-z14-0, formed roughly 290 million years after the Big Bang. The research team published their redshift measurements in Nature.", "Science"),
    ("CERN Physicists Measure Quantum Entanglement at the Highest Energy Scale Ever Recorded", 
     "Researchers at the ATLAS and CMS collaborations at CERN's Large Hadron Collider have observed quantum entanglement between top quarks and antiquarks at ultra-high energy scales. The peer-reviewed findings confirm Standard Model predictions.", "Science"),
    ("Perseverance Rover Collects Organic Rock Cores from Jezero Crater Delta on Mars", 
     "NASA's Mars 2020 Perseverance rover sealed its twenty-fourth geological sample core extracted from sedimentary river delta mudstones in Jezero Crater. Planetary scientists confirmed the presence of carbonates and silica minerals.", "Science"),
    ("European Space Agency Euclid Observatory Unveils First Detailed Map of Cosmic Web", 
     "The European Space Agency unveiled the first operational survey images captured by the Euclid space observatory, revealing cosmic filament structures across billions of light-years to map dark matter distribution.", "Science"),
    ("Astronomers Detect Water Vapor in Atmosphere of Habitable-Zone Exoplanet K2-18b", 
     "Using transmission spectroscopy data from the Hubble and James Webb Space Telescopes, astrophysicists identified atmospheric water vapor and carbon dioxide signatures surrounding sub-Neptune exoplanet K2-18b.", "Science"),
    ("Nuclear Fusion Facility Reaches Net Energy Gain Milestone in Controlled Plasma Reaction", 
     "Nuclear physicists at the National Ignition Facility announced that a laser-driven inertial confinement fusion experiment produced 3.8 megajoules of energy from an input of 2.05 megajoules, repeating target ignition.", "Science"),
    ("European Union Formally Approves Landmark Artificial Intelligence Act with Tiered Compliance", 
     "The Council of the European Union and European Parliament enacted the EU AI Act, establishing clear classification guidelines for high-risk biometric systems and transparency mandates for generative models.", "Technology"),
    ("MIT Engineers Develop Solid-State Lithium Battery Retaining 90 Percent Capacity Over 1000 Cycles", 
     "Materials scientists at MIT published a method in Nature Energy demonstrating a ceramic electrolyte interface that prevents metallic dendrite formation, enabling safe fast-charging electric vehicle batteries.", "Technology"),
    ("Geological Survey of India Confirms Substantial Lithium Exploration Reserves in Northern Belt", 
     "Geological Survey of India field teams published preliminary resource estimates indicating significant inferred lithium pegmatite deposits during commercial critical mineral survey programs.", "Geology & Resources"),
    ("Nobel Committee Awards Physics Prize for Advances in Attosecond Laser Pulse Generation", 
     "The Royal Swedish Academy of Sciences awarded the Nobel Prize in Physics to pioneering researchers who developed experimental methods generating attosecond pulses of light for studying electron dynamics.", "Science"),

    # Health, Medicine & Environment
    ("WHO Confirms Eradication of Wild Poliovirus Type 2 Across South-East Asia Region", 
     "The World Health Organization Regional Committee verified that sustained vaccination surveillance and community outreach successfully prevented wild poliovirus transmission across eleven countries.", "Health"),
    ("FDA Grants Approval for Novel CRISPR Gene Editing Therapy for Sickle Cell Disease", 
     "The United States Food and Drug Administration authorized the first cell-based gene therapy utilizing CRISPR/Cas9 technology to modify hematopoietic stem cells for patients suffering from severe sickle cell disease.", "Health"),
    ("Public Health Ministries Deliver Two Million Malaria Vaccine Doses in Sub-Saharan Africa", 
     "Routine childhood immunization initiatives completed delivery of over two million RTS,S malaria vaccine doses, resulting in a thirty percent decline in severe pediatric malaria hospitalizations.", "Health"),
    ("International Energy Agency Confirms Renewable Power Capacity Surpassed Coal Output", 
     "Annual power generation data compiled by the IEA showed that combined solar photovoltaic and wind installations produced more terawatt-hours of electricity worldwide than coal power plants during the prior calendar year.", "Environment"),
    ("Conservation Biologists Announce Reclassification of Iberian Lynx from Endangered to Vulnerable", 
     "The International Union for Conservation of Nature updated the red list status of the Iberian lynx after sustained habitat restoration expanded the wild breeding population to over 2,000 individuals.", "Environment"),
    ("Clinical Trial Demonstrates Targeted Immunotherapy Efficacy in Early-Stage Melanoma Patients", 
     "Oncologists published phase III trial results in the New England Journal of Medicine demonstrating that adjuvant checkpoint inhibitor therapy reduced cancer recurrence risks by forty percent in postoperative patients.", "Health & Medicine"),
    ("Indian Council of Medical Research Releases Standard Diagnostic Treatment Protocols for Vector Diseases", 
     "The Indian Council of Medical Research updated clinical diagnostic and fluid management guidelines for dengue and chikungunya cases admitted across primary health centers.", "Health"),
    ("United Nations Climate Conference Reaches Agreement on Global Loss and Damage Financial Facility", 
     "Delegates at the UN Climate Summit reached a consensus establishing an international loss and damage fund to assist developing nations vulnerable to extreme meteorological events.", "Environment & Policy")
]

FAKE_CORPUS = [
    # Viral Schemes, Cash Giveaways, Free Recharges & WhatsApp Forward Hoaxes
    ("Government Announces Scheme to Deposit Rs 10000 Monthly into Bank Accounts of All Citizens", 
     "A viral message circulating on social media and WhatsApp claims that the central government has announced a new nationwide welfare scheme under which every adult citizen will receive Rs 10,000 per month directly into their bank account starting next month. Click the unverified link to register your Aadhaar number before the portal closes!", "Welfare Hoax"),
    ("Prime Minister Free Laptop and Mobile Recharge Scheme Circulating on WhatsApp", 
     "BREAKING ANNOUNCEMENT! Government is distributing free high-end 5G smartphones, laptops, and 3 months of free mobile recharge to all students and citizens who forward this message to 10 WhatsApp groups and submit their bank details on the external form!", "Scheme Hoax"),
    ("All Old 500 Rupee Currency Notes Declared Invalid by Central Bank Starting Midnight", 
     "URGENT WARNING! Secret leaked circular claims that all existing 500 rupee notes will become completely illegal and banned from midnight tonight, ordering citizens to rush to banks immediately or lose their life savings! Forward this alert to everyone!", "Economy Hoax"),
    ("UNESCO Declares National Anthem Best and Most Patriotic Melody in the Entire World", 
     "GREAT PRIDE! UNESCO has officially announced in an international committee session that our national anthem has been declared the absolute best, most harmonious, and most patriotic anthem in the world! Share this proud news with everyone immediately!", "Culture Hoax"),
    ("Reserve Bank of India Installs GPS Nanochips in Currency Notes to Track Black Money from Satellites", 
     "BOMBSHELL TRUTH! New high-denomination currency notes contain microscopic radioactive nano-GPS chips that reflect signals to orbiting satellites, allowing tax officers to pinpoint hidden cash buried up to 120 meters underground without physical inspection! Share this before government deletes it!", "Economy Hoax"),
    ("Government Passes Emergency Order Confiscating Gold Jewelry Kept in Private Bank Lockers", 
     "URGENT WARNING! Secret circular leaked from finance ministry confirms that all gold and silver jewelry kept in bank locker vaults will be confiscated by authorities starting next Monday morning! Withdraw all your valuables before banks lock the doors!", "Economy Hoax"),
    ("Bank ATMs Across the Country Will Permanently Stop Dispensing Cash Starting Midnight Tonight", 
     "URGENT ALERT! Central banks and global elites have ordered every commercial automated teller machine disabled permanently starting at midnight tonight to enforce a mandatory cashless digital tyranny where citizen accounts are frozen without court orders! Withdraw all your cash right now!", "Economy Hoax"),
    ("WhatsApp Will Start Charging Monthly Subscription Fees Tomorrow Unless You Forward to Twenty Contacts", 
     "ATTENTION ALL USERS! WhatsApp will become a paid service costing 500 rupees per month from tomorrow morning! To keep your account completely free for life, you must forward this official message to at least 20 contacts immediately. Your logo will turn blue once activated!", "Social Media Hoax"),
    ("Government Distributing Free 5000 Rupee Festival Subsidy to All Family Ration Card Holders", 
     "HURRY UP! Under a new emergency festival relief scheme, government is transferring 5,000 rupees into bank accounts of all citizens. Fill in your bank account number and UPI PIN on the attached web link to claim before slots expire!", "Scam Hoax"),
    ("Central Government Banning All Private Cryptocurrency and Confiscating Digital Wallets Overnight", 
     "SECRET LEAKED ORDER! All personal cryptocurrency holdings and private wallet keys will be seized by federal agents starting tomorrow at midnight! Transfer your coins immediately to the safe recovery address provided below!", "Crypto Scam"),

    # Medical Hoaxes, Miracle Cures & Pseudo-Science
    ("Drinking Boiled Betel Leaves and Turmeric Completely Eliminates All Viral Diseases in Three Hours", 
     "AYURVEDIC MIRACLE DOCTORS SUPPRESSED! A viral message claims that boiling raw turmeric with betel leaf juice instantly kills all mutated viral strains in the human throat within 3 hours. Forward this urgent remedy to all family groups immediately!", "Health Hoax"),
    ("Drinking Raw Lemon Juice Mixed with Baking Soda Cures 100 Percent of Cancers Without Chemotherapy", 
     "Miracle cure exposed! Big Pharma executives and oncology hospitals are terrified of this one simple secret: drinking lemon juice steeped with baking soda destroys 100 percent of cancer tumors, leukemia, and viral diseases in 24 hours without chemotherapy! Share this miracle recipe!", "Health Hoax"),
    ("Drinking Hot Water and Lemon Kills Coronavirus and All Viral Diseases in Twenty Four Hours", 
     "DOCTORS ADMIT TRUTH! Drinking extremely hot water mixed with freshly squeezed lemon juice alters your body pH to alkaline, immediately dissolving viral coatings and curing respiratory infections in 24 hours. No doctors or hospitals needed!", "Health Hoax"),
    ("Eating Raw Garlic and Honey on Empty Stomach Completely Eliminates All Cancer Cells in Three Days", 
     "SECRET CANCER CURE REVEALED! Pharmaceutical companies are hiding this ancient herbal secret: eating crushed garlic cloves mixed with raw honey kills every single cancer cell and unclogs all heart arteries in just 3 days without surgery!", "Health Hoax"),
    ("Placing Sliced Red Onions in Socks While Sleeping Draws Out 100 Percent of Toxins Overnight", 
     "DOCTORS ARE SPEECHLESS! Putting raw red onion slices inside your socks before sleeping pulls out heavy metals, vaccine poisons, and dangerous parasites through foot pores, turning the onion jet black by morning! Try this bedroom trick tonight!", "Health Hoax"),
    ("Drinking Boiled Papaya Leaf Juice Overnight Permanently Cures Diabetes and Reverses Insulin Dependency", 
     "DIABETES CURED FOREVER! Big insulin manufacturers don't want you to discover that drinking concentrated papaya leaf extract overnight regenerates damaged pancreas cells and permanently cures type-2 diabetes in 12 hours!", "Health Hoax"),
    ("Microwave Oven Radiation Permanently Mutates Food DNA and Causes Instant Stomach Cancer", 
     "DEADLY RADIATION EXPOSED! Scientific whistleblowers reveal that microwave ovens emit high-frequency nuclear rays that destroy food molecular structures, turning wholesome meals into toxic carcinogenic poisons that rot human intestines instantly!", "Health Hoax"),
    ("Boiling Ginger with Cloves and Inhaling Steam Destroys Any Mutant Infection Instantly", 
     "HOME REMEDY MIRACLE! Never go to the hospital again! Inhaling steam from boiled ginger, cloves, and eucalyptus oil destroys 100% of lung bacteria and viral pathogens within 10 minutes. Share this life-saving secret with everyone!", "Health Hoax"),

    # Chemtrails, Space, 5G & Global Conspiracies
    ("Commercial Airplanes Secretly Spraying Mind Control Chemtrails to Alter Global Weather Patterns", 
     "SHOCKING PROOF! Whistleblower scientists have finally proven that commercial passenger airplanes are secretly spraying toxic chemtrails to control the weather, cause artificial droughts, and mind control the population! Leaked blueprints show hidden chemical tanks inside commercial airliner wings!", "Conspiracy"),
    ("NASA Confirms Earth Will Experience 15 Days of Total Pitch Darkness Next Month", 
     "BOMBSHELL REVELATION! Leaked internal NASA documents prove that our entire planet will be plunged into complete and absolute pitch darkness for fifteen consecutive days! Top scientists admit an alignment between Jupiter and Venus will spark cosmic explosions that blackout the sun completely! Share before deleted!", "Space Hoax"),
    ("James Webb Telescope Discovers Massive Artificial Alien Dyson Sphere Orbiting Nearby Star", 
     "Astronomers operating the James Webb Telescope were reportedly silenced after detecting a mega-engineering alien Dyson sphere blocking light from star HD 164595. Leaked telemetry logs purportedly prove extraterrestrial civilizations are harvesting solar energy!", "Space Hoax"),
    ("5G Cell Towers Emit Secret High-Frequency Radiation That Depletes Oxygen in Human Lungs", 
     "THE SHOCKING TRUTH! Electromagnetic radiation emitted by 5G mobile towers vibrates air molecules at 60GHz, preventing human hemoglobin from absorbing oxygen and causing sudden breathing collapse in major cities! Destroy local cell towers to save lives!", "Tech Hoax"),
    ("Government Mandates 5G Nano-Tracking Chips Injected into Commercial Bottled Drinking Water", 
     "EXPOSED! Whistleblower blueprints reveal that commercial water bottling corporations are secretly infusing microscopic radiofrequency identification nanochips into purified water bottles to monitor citizen locations and brain activity!", "Tech Hoax"),
    ("United Nations Passes Secret Treaty Banning All Gas Stoves and Lawn Mowers Worldwide Next Week", 
     "OUTRAGEOUS TYRANNY! In a midnight vote held behind locked doors, the UN General Assembly reportedly made it a criminal offense to own propane stoves or gasoline lawnmowers, ordering armed blue-helmet inspectors to conduct door-to-door confiscation raids!", "Politics Hoax"),
    ("Aliens Confirmed in Secret Underground Deep Space Bunker According to Leaked Pentagon Video", 
     "SHOCKING FOOTAGE LEAKED! Declassified military files reportedly confirm that government scientists have been harboring live extraterrestrial beings in deep subterranean hangars, suppressing advanced free zero-point energy generators from humanity!", "Conspiracy"),
    ("Secret Earthquake Weapon HAARP Caused Recent Natural Disasters Across Asia and Europe", 
     "EXPOSED! Former military officers confirm that ionospheric heating antennas deployed in secret polar research stations fired directed electromagnetic pulses into tectonic fault lines to trigger artificial earthquakes and tsunamis!", "Conspiracy")
]


def build_and_save_dataset(data_dir: str = "ml_model/data") -> pd.DataFrame:
    """
    Builds, enriches with both full articles and standalone headlines,
    and saves the verified fake news dataset.
    """
    os.makedirs(data_dir, exist_ok=True)
    records = []

    # Add real news (both combined text and standalone headlines for robust multi-length generalization)
    for title, text, category in REAL_CORPUS:
        combined = f"{title}. {text}"
        records.append({
            "title": title,
            "text": text,
            "combined_content": combined,
            "category": category,
            "label": REAL_LABEL,
            "label_name": "REAL",
            "source": "Verified News"
        })
        # Headline entry
        records.append({
            "title": title,
            "text": "",
            "combined_content": title,
            "category": category,
            "label": REAL_LABEL,
            "label_name": "REAL",
            "source": "Verified Headline"
        })

    # Add fake news (both combined text and standalone headlines)
    for title, text, category in FAKE_CORPUS:
        combined = f"{title}. {text}"
        records.append({
            "title": title,
            "text": text,
            "combined_content": combined,
            "category": category,
            "label": FAKE_LABEL,
            "label_name": "FAKE",
            "source": "Fact-Checked Hoax"
        })
        # Headline entry
        records.append({
            "title": title,
            "text": "",
            "combined_content": title,
            "category": category,
            "label": FAKE_LABEL,
            "label_name": "FAKE",
            "source": "Fact-Checked Hoax Headline"
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["combined_content"]).reset_index(drop=True)

    out_path = os.path.join(data_dir, "full_dataset.csv")
    df.to_csv(out_path, index=False)

    real_c = int((df["label"] == 0).sum())
    fake_c = int((df["label"] == 1).sum())
    print(f"Dataset Built: Total {len(df)} records (REAL: {real_c}, FAKE: {fake_c}) saved to {out_path}")
    return df


if __name__ == "__main__":
    build_and_save_dataset()
