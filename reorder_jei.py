"""Produce a JEI-ordered version of the paper (Methods moved to the end, per JEI
convention) and renumber sections + cross-references. Writes paper_jei.html."""
from pathlib import Path
HOME = Path.home()
src = (HOME / "paper" / "paper.html").read_text()

# Extract the Methods section (from its <h2> up to the Results <h2>) and move it
# to just before References.
m0 = src.index("<h2>3. Methods</h2>")
m1 = src.index("<h2>4. Results</h2>")
methods = src[m0:m1]
body = src[:m0] + src[m1:]                      # remove Methods from middle
ins = body.index("<h2>References</h2>")
body = body[:ins] + methods + body[ins:]        # reinsert before References

# Renumber headings (JEI order: 1 Intro, 2 Related Work, 3 Results, 4 Discussion,
# 5 Limitations, 6 Conclusion, 7 Methods) and fix §-cross-references.
for a, b in [
    ("<h2>3. Methods</h2>",     "<h2>7. Methods</h2>"),
    ("<h2>4. Results</h2>",     "<h2>3. Results</h2>"),
    ("<h2>5. Discussion</h2>",  "<h2>4. Discussion</h2>"),
    ("<h2>6. Limitations</h2>", "<h2>5. Limitations</h2>"),
    ("<h2>7. Conclusion</h2>",  "<h2>6. Conclusion</h2>"),
    ("<h3>4.", "<h3>3."),        # Results subsections 4.x -> 3.x
    ("§4.", "§3."),              # cross-references to Results
]:
    body = body.replace(a, b)

# Mark the variant in the running header.
body = body.replace(
    "Draft — generated from verified results",
    "Draft (JEI section order: Methods last) — generated from verified results")

(HOME / "paper" / "paper_jei.html").write_text(body)
print("wrote paper_jei.html")
