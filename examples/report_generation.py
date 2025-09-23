from franc.evaluation import report_generation as rg

r = rg.Report()

r["first title"] = {"section 1": ["abc", "def"]}
r["second title"] = [
    "abc",
    rg.ReportTable([["a", "b"], ["c", "d"]], ["col1", "col2"], "A table"),
]

r.compile("/tmp/test")
