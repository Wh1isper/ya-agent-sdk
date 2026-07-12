type MarkdownTreeNode = {
  type?: unknown
  depth?: unknown
  children?: unknown
}

function isMarkdownTreeNode(value: unknown): value is MarkdownTreeNode {
  return typeof value === 'object' && value !== null
}

/**
 * Keep model-generated Markdown beneath the route-owned h1 without allowing
 * a source document to skip heading levels. Source depth is shifted by one,
 * then clamped to at most one level deeper than the previous rendered heading.
 */
export function remarkNormalizeRouteHeadings() {
  return (tree: unknown) => {
    let previousDepth = 1

    const visit = (value: unknown) => {
      if (!isMarkdownTreeNode(value)) return
      if (
        value.type === 'heading' &&
        typeof value.depth === 'number' &&
        Number.isInteger(value.depth)
      ) {
        const shiftedDepth = Math.min(Math.max(value.depth + 1, 2), 6)
        const normalizedDepth = Math.min(shiftedDepth, previousDepth + 1)
        value.depth = normalizedDepth
        previousDepth = normalizedDepth
      }
      if (Array.isArray(value.children)) {
        value.children.forEach(visit)
      }
    }

    visit(tree)
  }
}
