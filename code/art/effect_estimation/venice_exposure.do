* Import solo dataset
import delimited "..\..\..\data\art\venice_matched.csv", clear

* Encode categorical variables and generate biennale year variable
encode gender_est, generate(gender_cat)
encode age_stage, generate(age_cat)
encode career_stage, generate(career_cat)
encode biennale_decade, generate(biennale_decade1)
gen new_year2biennale = year_diff + 5
xtset artist_id new_year2biennale

* Solo Analysis: Save ATET results
xthdidregress aipw (s f g b i.gender_cat europe asia northamerica southamerica oceania africa i.age_cat i.career_cat) (biennale_treated), group(artist_id)

parmest, saving(venice_exposure_solo_atet, replace)
use venice_exposure_solo_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.new_year2biennale")
gen to_year = cohort_number - 5
drop parm cohort_number eq
export delimited "..\..\..\results\art\venice_exposure_solo_atet.csv", replace

* Import dataset again for group analysis
import delimited "..\..\..\data\art\venice_matched.csv", clear

* Encode categorical variables and generate biennale year variable
encode gender_est, generate(gender_cat)
encode age_stage, generate(age_cat)
encode career_stage, generate(career_cat)
encode biennale_decade, generate(biennale_decade1)
gen new_year2biennale = year_diff + 5
xtset artist_id new_year2biennale

* Group Analysis: Save ATET results
xthdidregress aipw (g f s b i.gender_cat europe asia northamerica southamerica oceania africa i.age_cat i.career_cat) (biennale_treated), group(artist_id)

parmest, saving(venice_exposure_group_atet, replace)
use venice_exposure_group_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.new_year2biennale")
gen to_year = cohort_number - 5
drop parm cohort_number eq
export delimited "..\..\..\results\art\venice_exposure_group_atet.csv", replace

* Import dataset again for fair analysis
import delimited "..\..\..\data\art\venice_matched.csv", clear

* Encode categorical variables and generate biennale year variable
encode gender_est, generate(gender_cat)
encode age_stage, generate(age_cat)
encode career_stage, generate(career_cat)
encode biennale_decade, generate(biennale_decade1)
gen new_year2biennale = year_diff + 5
xtset artist_id new_year2biennale

* Fair Analysis: Save ATET results
xthdidregress aipw (f s g b i.gender_cat europe asia northamerica southamerica oceania africa i.age_cat i.career_cat) (biennale_treated), group(artist_id)

parmest, saving(venice_exposure_fair_atet, replace)
use venice_exposure_fair_atet.dta, clear
gen cohort_number = real(regexs(1)) if regexm(parm, "([0-9]+)\.new_year2biennale")
gen to_year = cohort_number - 5
drop parm cohort_number eq
export delimited "..\..\..\results\art\venice_exposure_fair_atet.csv", replace