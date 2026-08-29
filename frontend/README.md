# Triage Stream

Act as an expert UI/UX designer and React/Tailwind developer. Build a modern, clinical Triage Nurse Dashboard for an Emergency Department. 

Visual Theme: Clean, enterprise healthcare look. Strictly Light Mode (white and light-slate backgrounds) for high readability. 

Layout Structure: 

1. Left Sidebar (Fixed): 

- Hospital/App Logo at the top.

- Navigation links: Dashboard (active state), Patient Details, Settings.

- Logout button anchored at the bottom.

2. Top Header:

- Left: A clean Search Bar.

- Middle/Right (Hackathon Features): A "Kill AI / Manual Mode" toggle switch (default to AI Active/Green), and a "Simulate Surge" outline button.

- Far Right: User Profile photo/avatar.

3. Main Content Area (Scrollable):

- Stats Row: 4 clean summary cards at the top (e.g., Total Patients: 42, Active Staff: 8, Longest Wait: 45m, AI Agreement: 94%).

- Queue Header: In a single horizontal line, place a bold heading "Patient Queue" on the left, and a prominent primary CTA button "+ Add Patient" on the right.

- The Patient Queue (Horizontal Stack / Table List):

Create a list view (NOT side-by-side grid cards, but horizontal rows stacked vertically). The mock data must be sorted by ESI Priority (1 to 5) first, then by Wait Time.

Columns required:

  * Patient MRN / Name (e.g., "Doe, John | MRN-A3F9")

  * Age & Gender (e.g., "45 M")

  * Chief Complaint Summary

  * Assigned ESI Score (Render as a colored badge: ESI 1=Red, ESI 2=Orange, ESI 3=Yellow, ESI 4=Green, ESI 5=Blue).

  * Current Wait Time (e.g., "14 mins")

*CRITICAL FEATURE*: Make one of the ESI 2 or 3 rows look "alerted" (e.g., a light red tinted background or a ⚠️ "Needs Reassessment" tag) to demonstrate our background deterioration monitor catching a patient who has waited too long.

- Bottom Section / Analytics:

On the bottom right of the main content area, include a sleek Radial/Donut chart component titled "Current ESI Distribution". It should show a breakdown of the number of patients currently assigned to each ESI score (1 through 5), using the corresponding ESI colors.

u can use the reference image, but make sure the queue is horizontally stacked and not side by side

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/56b503ab-e73e-42cd-9a8e-4ec82d281a7a).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
