import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen year_to_publish = to_year + 5
keep if gender ==1
encode career_stage, generate(career_cat)
xtset researcher_id year_to_publish

* Run xthdidregress command
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_cit_atet_f.csv", replace

import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen year_to_publish = to_year + 5
keep if gender ==0
encode career_stage, generate(career_cat)
xtset researcher_id year_to_publish

* Run xthdidregress command
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_cit_atet_m.csv", replace