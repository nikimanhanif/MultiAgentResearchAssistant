// Helper to detect if a message contains a research report
export function isReportMessage(content: string): boolean {
  const reportIndicators = [
    '# ',
    '## Executive Summary',
    '## Key Findings',
    '## Introduction',
    '## Literature Review',
    '## Methodology',
    '## Conclusion',
    '## References'
  ]

  const hasMultipleHeaders = (content.match(/^##?\s/gm) || []).length >= 2
  const hasReportIndicator = reportIndicators.some(indicator => content.includes(indicator))
  const isLongEnough = content.length > 500

  return hasMultipleHeaders && (hasReportIndicator || isLongEnough)
}
