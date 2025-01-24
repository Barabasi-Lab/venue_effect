import delimited "..\..\..\data\science\pnas_matched.csv", clear 
gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)
xtset researcher_id year

* Run xthdidregress command
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)

* Save ATET results using parmest
parmest, saving(pnas_exposure_cit_atet, replace)
use pnas_exposure_cit_atet.dta, clear

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\pnas_variation_cit_atet.csv", replace

import delimited "..\..\..\data\science\pnas_matched.csv", clear 
gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)
xtset researcher_id year
* Run xthdidregress command
xthdidregress ra (cum_publication_count cum_citations_na cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)

* Save ATET results using parmest
parmest, saving(pnas_exposure_prod_atet, replace)
use pnas_exposure_prod_atet.dta, clear

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\pnas_variation_prod_atet.csv", replace