# Authlab Attendance Report Generator

Generate graphical attendance reports from Authlab attendance data.

## Setup

### 1. Get Attendance Data from API

1. Open [Authlab](https://lounge.authlab.io) in your browser
2. Navigate to **HR Home** → **Attendances**
3. Open browser **Developer Tools** (F12 or Right-click → Inspect)
4. Go to the **Network** tab
5. Look for API request: `my-attendances?per_page=150` (or similar)
6. Click on the request and go to **Response** tab
7. Copy the **`data`** array only (not the entire response)
8. Save it to `authlab_attendences.json` in this format:

```json
{
  "data": [
    // paste your copied data array here
  ]
}
```

> **Tip:** If you have more than 150 records, increase `per_page` parameter in the API URL or paginate through multiple pages.

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the Script

```bash
python3 generate_attendance_report.py
```

The report will be generated in the `report/` directory and automatically opened in your default browser.

## Output

The script generates:

- `report/attendance_report.html` - Main interactive report
- `report/*.png` - Static chart images
- `report/*.html` - Interactive Plotly charts

## Features

- **Daily hours tracking** with 9-hour target line
- **Monthly/Yearly breakdowns**
- **Arrival/Exit time distributions**
- **Day-of-week analysis**
- **Duration categories** (9h+, 10h+, 11h+, 12h+)
- **Weekend work tracking**
- **Year-by-year comparison**

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies
