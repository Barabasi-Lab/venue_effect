* Early career + citations
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 10
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "early-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_cit_atet_early.csv", replace

* Early career + publications
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 30
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "early-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_publication_count cum_citations_na cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_prod_atet_early.csv", replace




* Mid career + citations
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 30
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "mid-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_cit_atet_mid.csv", replace

* Mid career + publications
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 30
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "mid-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_publication_count cum_citations_na cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_prod_atet_mid.csv", replace




* Late career + citations
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 30
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "late-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_citations_na cum_publication_count cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_cit_atet_late.csv", replace

* Late career + publications
import delimited "..\..\..\data\science\nature_matched.csv", clear 
gen new_career_stage = "" 
replace new_career_stage = "early-career" if (first_publish_year - first_year) <= 10
replace new_career_stage = "mid-career" if (first_publish_year - first_year) > 10 & (first_publish_year - first_year) <= 30
replace new_career_stage = "late-career" if (first_publish_year - first_year) > 30

gen str venue_career_stage = ""

bysort researcher_id (to_year): replace venue_career_stage = new_career_stage if to_year == 0

* Propagate the value forwards within each researcher_id
bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n-1] if missing(venue_career_stage)
* Propagate the value backwards within each researcher_id using a loop
local max_loops = 5 
forval i = 1/`max_loops' {
    bysort researcher_id (to_year): replace venue_career_stage = venue_career_stage[_n+1] if missing(venue_career_stage)
}

* Filter the dataset to include only rows where 'venue_career_stage' matches a specific condition
keep if venue_career_stage == "late-career"

gen year_to_publish = to_year + 5
encode career_stage, generate(career_cat)

xtset researcher_id year_to_publish
xthdidregress ra (cum_publication_count cum_citations_na cum_funding_count cum_corresponding_count i.career_cat) (is_journal), group(researcher_id)
estat atetplot

* Save ATET results using parmest
parmest, saving(nature_exposure_cit_atet, replace)
use nature_exposure_cit_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.year_to_publish")
gen to_year = cohort_number - 5
drop parm cohort_number eq

* Save the modified dataset as a CSV file
export delimited "..\..\..\results\science\nature_exposure_prod_atet_late.csv", replace