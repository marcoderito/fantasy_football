"""
PDF Report Generator

Generates a comprehensive PDF report for a multi-manager fantasy football auction:
  1. Cover page (title, author, course info)
  2. Index page for quick navigation
  3. Technical overview and scoring function details
  4. Empirical performance analysis (tables + charts)
  5. Forced assignments and budget usage stats
  6. Final team listings and best manager highlight
  7. Conclusions and disclaimers

Assumptions:
 - Each Manager object provides:
    manager.name (str)
    manager.strategy (str)
    manager.team (List[Player]) with player.final_price (float)
    manager.budget (float) left after auction
    manager.forced_assignments (List[Player]) if forced picks occurred

Author: Marco De Rito
"""

import os
import statistics
import textwrap
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    from utils import score_player
except ImportError:
    def score_player(_unused_player):
        return 0.0


def generate_pdf_report(managers, filename="report.pdf"):
    """
    Creates a multi-page PDF report with:
      - Cover page
      - Index page
      - Technical overview (algorithms, scoring)
      - Performance data: aggregated stats, charts
      - Forced assignments & budget usage
      - Final teams and highlight of best manager
      - Conclusions and disclaimers
    """
    # Attempt to register a custom font, fallback to 'Helvetica' if unavailable
    try:
        pdfmetrics.registerFont(TTFont('HelveticaNeue', 'HelveticaNeue.ttf'))
        main_font = "HelveticaNeue"
    except Exception:
        main_font = "Helvetica"

    # Collect stats for analysis
    performance_data = {}
    manager_distribution = {}
    all_player_scores = []
    best_manager_name = None
    best_manager_score = -1

    # Arrays for forced and budget stats
    managers_names = []
    forced_counts = []
    budget_spent_values = []
    leftover_values = []
    final_scores = []

    for mgr in managers:
        strategy = mgr.strategy.upper()
        team_scores = [score_player(p) for p in mgr.team]
        total_score = sum(team_scores)
        avg_score = (total_score / len(team_scores)) if team_scores else 0.0

        if strategy not in performance_data:
            performance_data[strategy] = {
                "managers": 0,
                "total_score": 0.0,
                "avg_team_score": 0.0
            }
        performance_data[strategy]["managers"] += 1
        performance_data[strategy]["total_score"] += total_score
        performance_data[strategy]["avg_team_score"] += avg_score

        manager_distribution[strategy] = manager_distribution.get(strategy, 0) + 1
        all_player_scores.extend(team_scores)

        if total_score > best_manager_score:
            best_manager_score = total_score
            best_manager_name = mgr.name

        total_spent = sum(p.final_price for p in mgr.team)
        forced_list = getattr(mgr, 'forced_assignments', [])

        managers_names.append(mgr.name)
        forced_counts.append(len(forced_list))
        budget_spent_values.append(total_spent)
        leftover_values.append(mgr.budget)
        final_scores.append(total_score)

    # Create a table for strategy-based performance
    performance_table = []
    table_header = ["Strategy", "Managers", "Avg Total Score", "Avg Team Score"]
    performance_table.append(table_header)

    for strat, vals in performance_data.items():
        count = vals["managers"]
        avg_total = (vals["total_score"] / count) if count else 0.0
        avg_team = (vals["avg_team_score"] / count) if count else 0.0
        performance_table.append([
            strat,
            str(count),
            f"{avg_total:.2f}",
            f"{avg_team:.2f}"
        ])

    # Overall player stats (for histogram)
    if all_player_scores:
        best_player_score = max(all_player_scores)
        worst_player_score = min(all_player_scores)
        avg_player_score = statistics.mean(all_player_scores)
    else:
        best_player_score = worst_player_score = avg_player_score = 0.0

    # Generate charts
    plt.figure()
    strategies = [row[0] for row in performance_table[1:]]
    avg_team_scores = [float(row[3]) for row in performance_table[1:]]
    plt.bar(strategies, avg_team_scores, color='skyblue', edgecolor='black')
    plt.xlabel("Strategy")
    plt.ylabel("Avg Team Score")
    plt.title("Average Team Score by Strategy")
    chart_avg_score = "temp_avg_score.png"
    plt.savefig(chart_avg_score, bbox_inches='tight')
    plt.close()

    plt.figure()
    labels = list(manager_distribution.keys())
    sizes = list(manager_distribution.values())
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Manager Distribution by Strategy")
    chart_managers_pie = "temp_manager_pie.png"
    plt.savefig(chart_managers_pie, bbox_inches='tight')
    plt.close()

    plt.figure()
    if all_player_scores:
        plt.hist(all_player_scores, bins=10, color='lightgreen', edgecolor='black')
        plt.title("Distribution of Player Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
    else:
        plt.text(0.5, 0.5, "No player scores available", ha='center', va='center')
    chart_hist_scores = "temp_hist_scores.png"
    plt.savefig(chart_hist_scores, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(managers_names, forced_counts, color='orange', edgecolor='black')
    plt.title("Forced Assignments per Manager")
    plt.xlabel("Manager")
    plt.ylabel("Forced Players")
    plt.xticks(rotation=45, ha='right')
    chart_forced_assign = "temp_forced_assign.png"
    plt.tight_layout()
    plt.savefig(chart_forced_assign, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 4))
    bottom_vals = [0]*len(managers_names)
    plt.bar(managers_names, budget_spent_values, color='orchid', edgecolor='black', label='Spent')
    plt.bar(managers_names, leftover_values, bottom=budget_spent_values,
            color='lightgray', edgecolor='black', label='Leftover')
    plt.title("Budget Spent vs. Leftover")
    plt.ylabel("Budget Credits")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    chart_budget_spent = "temp_budget_spent.png"
    plt.tight_layout()
    plt.savefig(chart_budget_spent, bbox_inches='tight')
    plt.close()

    # Build the PDF
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    def new_page():
        c.showPage()
        return height - margin

    # Cover page
    c.setFont(main_font, 24)
    c.drawCentredString(width/2, y, "Fantasy Football Optimization Report")
    y -= 50
    c.setFont(main_font, 14)
    c.drawCentredString(width/2, y, "An Evolutionary & Swarm Intelligence Approach")
    y -= 20
    c.setFont(main_font, 10)
    c.drawCentredString(width/2, y, "Author: Marco De Rito")
    y -= 15
    c.drawCentredString(width/2, y, "University of Example - Advanced Optimization Course")
    y -= 80
    c.setFont(main_font, 12)
    c.drawCentredString(width/2, y, "Proceed to next page for index and details.")
    c.showPage()

    # Index page
    y = height - margin
    c.setFont(main_font, 18)
    c.drawString(margin, y, "Index")
    y -= 30
    c.setFont(main_font, 11)
    index_points = [
        "1. Technical Overview & Scoring",
        "2. Empirical Performance Analysis",
        "3. Forced Assignments & Budget Usage",
        "4. Final Teams",
        "5. Conclusions & Disclaimer",
    ]
    for pt in index_points:
        c.drawString(margin, y, f"- {pt}")
        y -= 15
    y -= 20
    c.setFont(main_font, 10)
    index_text = (
        "This index provides a quick guide through the main sections of the report. "
        "Page numbers are not explicitly shown, but each section title is clear."
    )
    wrapped_idx = textwrap.wrap(index_text, width=80)
    txt_idx = c.beginText(margin, y)
    txt_idx.setLeading(14)
    for line in wrapped_idx:
        txt_idx.textLine(line)
    c.drawText(txt_idx)
    c.showPage()

    # 1. Technical Overview & Scoring
    y = height - margin
    c.setFont(main_font, 16)
    c.drawString(margin, y, "1. Technical Overview & Scoring")
    y -= 30
    c.setFont(main_font, 10)

    tech_paragraph = [
        "This project combines PSO, DE, and ES to solve a multi-manager fantasy football optimization. "
        "Each manager chooses one approach and aims to maximize the total performance score of selected "
        "players under budget and positional constraints.",
        "",
        "Scoring Function (score_player):",
        "Rewards goals, assists, and bonus metrics, while penalizing negative events such as cards. "
        "A final fantasy rating may also contribute. The team's objective is the sum of these scores.",
        "",
        "Auction & Forced Assignments:",
        "Managers bid in parallel. If conflicts arise, a resolution decides who wins. If a manager fails "
        "to fulfill minimum role requirements or runs low on budget, forced assignments ensure a valid "
        "roster, but may not be optimal."
    ]
    txt_tech = c.beginText(margin, y)
    txt_tech.setLeading(14)
    for paragraph in tech_paragraph:
        wrapped_tech = textwrap.wrap(paragraph, width=95)
        for line in wrapped_tech:
            txt_tech.textLine(line)
        txt_tech.textLine("")
    c.drawText(txt_tech)
    y = txt_tech.getY() - 20

    # 2. Empirical Performance
    if y < 100:
        y = new_page()
    c.setFont(main_font, 16)
    c.drawString(margin, y, "2. Empirical Performance Analysis")
    y -= 30
    c.setFont(main_font, 10)
    c.drawString(margin, y, "Performance by Strategy:")
    y -= 20

    row_height = 14
    col_widths = [100, 60, 100, 100]
    x_start = margin

    for row_data in performance_table:
        x = x_start
        for i, cell in enumerate(row_data):
            c.drawString(x, y, cell)
            x += col_widths[i]
        y -= row_height
        if y < 80:
            y = new_page()

    chart_title = "Figure 1: Average Team Score by Strategy"
    c.drawString(margin, y, chart_title)
    y -= 15
    img_w, img_h = 300, 220
    if y - img_h < margin:
        y = new_page()
    c.drawImage(chart_avg_score, margin, y - img_h, width=img_w, height=img_h)
    y -= (img_h + 25)

    if y < 280:
        y = new_page()
    c.drawString(margin, y, "Figure 2: Manager Distribution by Strategy")
    y -= 15
    c.drawImage(chart_managers_pie, margin, y - img_h, width=img_w, height=img_h)
    y -= (img_h + 25)

    if y < 280:
        y = new_page()
    c.drawString(margin, y, "Figure 3: Distribution of Player Scores")
    y -= 15
    c.drawImage(chart_hist_scores, margin, y - img_h, width=img_w, height=img_h)
    y -= (img_h + 25)

    summary_text = (
        f"Player Score Summary:\n"
        f" - Best: {best_player_score:.2f}\n"
        f" - Worst: {worst_player_score:.2f}\n"
        f" - Average: {avg_player_score:.2f}"
    )
    txt_sum = c.beginText(margin, y)
    txt_sum.setLeading(14)
    for line in summary_text.split('\n'):
        txt_sum.textLine(line)
    c.drawText(txt_sum)
    y = txt_sum.getY() - 20

    # 3. Forced Assignments & Budget Usage
    if y < 250:
        y = new_page()
    c.setFont(main_font, 16)
    c.drawString(margin, y, "3. Forced Assignments & Budget Usage")
    y -= 30

    c.setFont(main_font, 10)
    c.drawString(margin, y, "Figure 4: Forced Assignments per Manager")
    y -= 15
    if y - img_h < margin:
        y = new_page()
    c.drawImage(chart_forced_assign, margin, y - img_h, width=img_w, height=img_h)
    y -= (img_h + 25)

    if y < 280:
        y = new_page()
    c.drawString(margin, y, "Figure 5: Budget Spent vs. Leftover")
    y -= 15
    c.drawImage(chart_budget_spent, margin, y - img_h, width=img_w, height=img_h)
    y -= (img_h + 30)

    c.drawString(margin, y, "Recap Table: Manager Stats")
    y -= 15
    recap_header = ["Manager", "Strat", "Forced", "Spent", "Leftover", "Objective"]
    recap_rows = [recap_header]
    for i, mgr_name in enumerate(managers_names):
        s_forced = forced_counts[i]
        s_spent = f"{budget_spent_values[i]:.1f}"
        s_left = f"{leftover_values[i]:.1f}"
        s_score = f"{final_scores[i]:.2f}"
        recap_rows.append([
            mgr_name, managers[i].strategy.upper(),
            str(s_forced), s_spent, s_left, s_score
        ])

    row_height = 14
    col_widths = [100, 50, 50, 60, 60, 70]
    x_start = margin
    for row_data in recap_rows:
        x = x_start
        for idx, cell_data in enumerate(row_data):
            c.drawString(x, y, cell_data)
            x += col_widths[idx]
        y -= row_height
        if y < 80:
            y = new_page()

    # 4. Final Teams
    y = new_page()
    c.setFont(main_font, 16)
    c.drawString(margin, y, "4. Final Teams")
    y -= 30

    c.setFont(main_font, 11)
    best_line = (
        f"The best manager overall is {best_manager_name} "
        f"with a total score of {best_manager_score:.2f}."
    )
    c.drawString(margin, y, best_line)
    y -= 25

    for mgr in managers:
        c.setFont(main_font, 12)
        m_header = f"{mgr.name} ({mgr.strategy.upper()})"
        if mgr.name == best_manager_name:
            m_header += "  <-- Top Performer"
        c.drawString(margin, y, m_header)
        y -= 20

        c.setFont(main_font, 10)
        for player in mgr.team:
            line_str = (
                f"- {player.name} ({player.role}), Price: {player.final_price:.1f}, "
                f"Score: {score_player(player):.2f}"
            )
            wrapped_line = textwrap.wrap(line_str, width=90)
            for wline in wrapped_line:
                c.drawString(margin+15, y, wline)
                y -= 12
                if y < 70:
                    y = new_page()
        y -= 10
        if y < 70:
            y = new_page()

    # 5. Conclusions & Disclaimer
    y = new_page()
    c.setFont(main_font, 16)
    c.drawString(margin, y, "5. Conclusions & Disclaimer")
    y -= 30
    c.setFont(main_font, 10)
    conclusion_lines = [
        "Conclusions:",
        " - Each manager's strategy aims to maximize total score under strict budget and role constraints.",
        " - Forced assignments ensure minimal positional requirements are met when normal bidding fails.",
        " - Final stats show how each manager allocated their credits, how many forced players they took,",
        "   and their ultimate objective score.",
        "",
        "Disclaimer:",
        "This report is provided for educational demonstration of swarm and evolutionary algorithms ",
        "in a multi-manager fantasy football auction scenario. For real-world usage, further validation ",
        "and domain expertise may be required.",
        "",
        "End of Report."
    ]
    txt_concl = c.beginText(margin, y)
    txt_concl.setLeading(14)
    for para in conclusion_lines:
        wrap_conc = textwrap.wrap(para, width=95)
        for wline in wrap_conc:
            txt_concl.textLine(wline)
        txt_concl.textLine("")
    c.drawText(txt_concl)

    c.save()

    # Cleanup temporary chart files
    for tmp_file in [
        chart_avg_score, chart_managers_pie, chart_hist_scores,
        chart_forced_assign, chart_budget_spent
    ]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


if __name__ == "__main__":
    class DummyPlayer:
        def __init__(self, name, role, final_price):
            self.name = name
            self.role = role
            self.final_price = final_price

    class DummyManager:
        def __init__(self, name, strategy, team, budget=0, forced_assignments=None):
            self.name = name
            self.strategy = strategy
            self.team = team
            self.budget = budget
            self.forced_assignments = forced_assignments or []

    # Basic test usage
    mgr1 = DummyManager(
        "Manager_Alpha", "pso",
        [
            DummyPlayer("Player A", "A", 15.0),
            DummyPlayer("Player B", "D", 10.0),
        ],
        budget=5.0,
        forced_assignments=[DummyPlayer("Player_C", "C", 1.0)]
    )

    mgr2 = DummyManager(
        "Manager_Beta", "de",
        [
            DummyPlayer("Player X", "C", 12.0),
            DummyPlayer("Player Y", "A", 17.0),
        ],
        budget=3.0
    )

    generate_pdf_report([mgr1, mgr2], filename="improved_report.pdf")
    print("PDF generated: improved_report.pdf")
